"""Real modality-level ranking tests for the UniIR and SpeechCLIP bridges.

Runs live HTTP queries against the deployed GPU services and scores them with
Recall@1/5/10 (UniIR-style hit rate), MRR, and latency per modality:

    UniIR:   text -> image, image -> image, image+text -> image
    SpeechCLIP: speech -> image, speech+image -> image

Usage:
    python -m competitors.run_ranking_tests \
        --uniir-url https://<uniir>.modal.run \
        --speechclip-url https://<speechclip>.modal.run \
        --tts

Speech queries are synthesized from ABO product names with gTTS (--tts) or read
from --audio-dir as {item_id}.wav files. Ground truth for every query is the
product the query was derived from, so the metrics describe zero-shot
retrieval on the shared ABO fixture catalog.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from competitors.common.abo_catalog import catalog_version, download_images, image_url, load_catalog
from competitors.common.ranking_metrics import aggregate, mrr, recall_at_k

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CATEGORY_SINGULAR = {"Sofas": "sofa", "Chairs": "chair", "Tables": "table", "Storage": "cabinet", "Beds": "bed"}


def multipart_body(fields: Dict[str, str], files: Dict[str, tuple]) -> tuple[bytes, str]:
    boundary = f"----vibe{uuid.uuid4().hex}"
    lines: List[bytes] = []
    for name, value in fields.items():
        lines.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())
    for name, (filename, content, content_type) in files.items():
        lines.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\n"
            f"Content-Type: {content_type}\r\n\r\n".encode()
            + content
            + b"\r\n"
        )
    lines.append(f"--{boundary}--\r\n".encode())
    return b"".join(lines), f"multipart/form-data; boundary={boundary}"


def post_search(url: str, fields: Dict[str, str], files: Dict[str, tuple], timeout: int = 120) -> tuple[Dict[str, Any], float]:
    body, content_type = multipart_body(fields, files)
    request = urllib.request.Request(url.rstrip("/") + "/search", data=body, headers={"Content-Type": content_type})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    latency_ms = (time.perf_counter() - started) * 1000
    return payload, latency_ms


def score_query(ranked: List[str], relevant: List[str], latency_ms: float, ks=(1, 5, 10)) -> Dict[str, Any]:
    row: Dict[str, Any] = {f"recall@{k}": recall_at_k(ranked, relevant, k) for k in ks}
    row["mrr"] = mrr(ranked, relevant)
    row["latency_ms"] = round(latency_ms, 1)
    return row


def synth_speech(text: str, out_path: Path) -> Optional[Path]:
    try:
        from gtts import gTTS
    except ImportError:
        print("gTTS not installed; skipping TTS speech queries (pip install gTTS)")
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        gTTS(text=text, lang="en").save(str(out_path))
    return out_path


def read_file(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def run_uniir_tests(url: str, products: List[Dict[str, Any]], image_paths: Dict[str, str], limit: int, top_k: int) -> Dict[str, Any]:
    modalities: Dict[str, List[Dict[str, Any]]] = {"text2image": [], "image2image": [], "image_text2image": []}
    sample = products[:limit]
    for product in sample:
        relevant = [product["id"]]
        color = (product.get("color") or "").strip()
        category = CATEGORY_SINGULAR.get(product.get("category"), (product.get("category") or "furniture").lower().rstrip("s"))
        material = (product.get("material") or "").strip()
        text_query = " ".join(part for part in [color, material, category] if part) or product["title"]

        payload, latency = post_search(url, {"query_txt": text_query, "candidate_modality": "image", "top_k": str(top_k)}, {})
        ranked = [r["product_id"] for r in payload.get("results", [])]
        modalities["text2image"].append({"product_id": product["id"], "query": text_query, **score_query(ranked, relevant, latency)})

        image_path = image_paths.get(product["id"])
        if image_path:
            image_file = ("image.jpg", read_file(image_path), "image/jpeg")
            payload, latency = post_search(url, {"candidate_modality": "image", "top_k": str(top_k)}, {"query_image": image_file})
            ranked = [r["product_id"] for r in payload.get("results", [])]
            modalities["image2image"].append({"product_id": product["id"], **score_query(ranked, relevant, latency)})

            payload, latency = post_search(
                url,
                {"query_txt": text_query, "candidate_modality": "image", "top_k": str(top_k)},
                {"query_image": image_file},
            )
            ranked = [r["product_id"] for r in payload.get("results", [])]
            modalities["image_text2image"].append({"product_id": product["id"], **score_query(ranked, relevant, latency)})

    return {name: aggregate(rows) for name, rows in modalities.items()}


def run_speechclip_tests(
    url: str,
    products: List[Dict[str, Any]],
    image_paths: Dict[str, str],
    audio_dir: Path,
    use_tts: bool,
    limit: int,
    top_k: int,
) -> Dict[str, Any]:
    modalities: Dict[str, List[Dict[str, Any]]] = {"speech2image": [], "speech_image2image": []}
    sample = products[:limit]
    for product in sample:
        relevant = [product["id"]]
        audio_path = audio_dir / f"{product['id']}.wav"
        if use_tts and not audio_path.exists():
            audio_path = synth_speech(product["title"], audio_dir / f"{product['id']}.wav") or audio_path
        if not audio_path.exists():
            continue

        audio_file = (audio_path.name, read_file(str(audio_path)), "audio/wav")
        payload, latency = post_search(url, {"candidate_modality": "image", "top_k": str(top_k)}, {"audio": audio_file})
        ranked = [r["product_id"] for r in payload.get("results", [])]
        modalities["speech2image"].append({"product_id": product["id"], **score_query(ranked, relevant, latency)})

        image_path = image_paths.get(product["id"])
        if image_path:
            payload, latency = post_search(
                url,
                {"candidate_modality": "image", "top_k": str(top_k)},
                {"audio": audio_file, "query_image": ("image.jpg", read_file(image_path), "image/jpeg")},
            )
            ranked = [r["product_id"] for r in payload.get("results", [])]
            modalities["speech_image2image"].append({"product_id": product["id"], **score_query(ranked, relevant, latency)})

    return {name: aggregate(rows) for name, rows in modalities.items()}


def write_report(results: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    json_path = out_dir / f"ranking-{stamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    lines = ["# Modality-level ranking results", "", f"Generated: {results['generated_at']}", ""]
    for provider, modalities in results["providers"].items():
        lines.append(f"## {provider}")
        lines.append("")
        lines.append("| modality | queries | recall@1 | recall@5 | recall@10 | mrr | mean latency ms |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for modality, summary in modalities.items():
            lines.append(
                f"| {modality} | {int(summary.get('queries', 0))} | {summary.get('recall@1', 0):.3f} | "
                f"{summary.get('recall@5', 0):.3f} | {summary.get('recall@10', 0):.3f} | "
                f"{summary.get('mrr', 0):.3f} | {summary.get('mean_latency_ms', 0):.0f} |"
            )
        lines.append("")
    md_path = out_dir / f"ranking-{stamp}.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {json_path} and {md_path}")
    return md_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uniir-url", default=os.environ.get("UNIIR_API_URL", ""))
    parser.add_argument("--speechclip-url", default=os.environ.get("SPEECHCLIP_API_URL", ""))
    parser.add_argument("--catalog", default="competitors/data/abo-catalog.json")
    parser.add_argument("--image-cache", default="competitors/data/abo-images-test")
    parser.add_argument("--audio-dir", default="competitors/data/test-audio")
    parser.add_argument("--limit", type=int, default=25, help="products per modality")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--tts", action="store_true", help="synthesize speech queries with gTTS")
    args = parser.parse_args()

    if not args.uniir_url and not args.speechclip_url:
        raise SystemExit("Set --uniir-url / --speechclip-url or UNIIR_API_URL / SPEECHCLIP_API_URL")

    catalog = load_catalog(args.catalog)
    products = catalog["products"]
    image_paths = download_images(products, args.image_cache)
    print(f"Catalog {catalog_version(catalog)}: {len(products)} products, {len(image_paths)} images cached")

    results: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "catalog_version": catalog_version(catalog),
        "top_k": args.top_k,
        "providers": {},
    }

    if args.uniir_url:
        print(f"Testing UniIR at {args.uniir_url}")
        results["providers"]["uniir"] = run_uniir_tests(args.uniir_url, products, image_paths, args.limit, args.top_k)
        for modality, summary in results["providers"]["uniir"].items():
            print(f"  uniir {modality}: {summary}")

    if args.speechclip_url:
        print(f"Testing SpeechCLIP at {args.speechclip_url}")
        results["providers"]["speechclip"] = run_speechclip_tests(
            args.speechclip_url, products, image_paths, Path(args.audio_dir), args.tts, args.limit, args.top_k
        )
        for modality, summary in results["providers"]["speechclip"].items():
            print(f"  speechclip {modality}: {summary}")

    write_report(results, RESULTS_DIR)


if __name__ == "__main__":
    main()
