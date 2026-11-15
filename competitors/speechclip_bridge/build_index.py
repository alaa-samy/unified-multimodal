"""Build the SpeechCLIP-specific ABO FAISS index.

Embeds every ABO candidate image with the checkpoint's frozen CLIP image
encoder (the same space SpeechCLIP speech embeddings are aligned to) and writes
indexes/speechclip/. This index is separate from the UniIR index: UniIR uses
its fine-tuned CLIP-SF weights, SpeechCLIP uses the checkpoint's own CLIP.

Local run:
    python -m competitors.speechclip_bridge.build_index --checkpoint /path/speechclip.ckpt

On Modal (GPU):
    modal run competitors/speechclip_bridge/modal_app.py::build-index
"""
from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
from PIL import Image

from competitors.common.abo_catalog import catalog_version, download_images, image_url, load_catalog
from competitors.common.faiss_index import build_index, save_index
from competitors.speechclip_bridge.model import SpeechCLIPWrapper, download_checkpoint


def load_wav_16k(path: str, target_sr: int = 16000) -> np.ndarray:
    import librosa

    wav, _ = librosa.load(path, sr=target_sr, mono=True)
    return wav


def build_speechclip_abo_index(
    catalog_path: str,
    out_dir: str,
    checkpoint_path: str,
    image_cache_dir: str = "competitors/data/abo-images-speechclip",
    batch_size: int = 16,
) -> Dict[str, object]:
    catalog = load_catalog(catalog_path)
    products = catalog["products"]
    with_images = [p for p in products if image_url(p)]
    print(f"Embedding {len(with_images)} ABO images with the SpeechCLIP checkpoint CLIP encoder")

    image_paths = download_images(with_images, image_cache_dir)
    model = SpeechCLIPWrapper(checkpoint_path=checkpoint_path)

    ready = [(p, image_paths[p["id"]]) for p in with_images if p["id"] in image_paths]
    embeddings = model.encode_images([Image.open(path) for _, path in ready], batch_size=batch_size)
    product_ids: List[str] = [p["id"] for p, _ in ready]

    index = build_index(embeddings, product_ids)
    out_path = save_index(
        index,
        product_ids,
        {
            "provider": "speechclip",
            "model": "SpeechCLIP parallel large (Flickr8k), frozen CLIP image encoder",
            "checkpoint": str(checkpoint_path),
            "catalog_version": catalog_version(catalog),
            "modality": "image candidates, speech (+optional image) queries",
        },
        out_dir,
    )
    skipped = len(products) - len(ready)
    print(f"SpeechCLIP ABO index: {index.ntotal} vectors, dim {index.d}, saved to {out_path} (skipped {skipped})")
    return {"size": int(index.ntotal), "dim": int(index.d), "path": str(out_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="competitors/data/abo-catalog.json")
    parser.add_argument("--out", default="indexes/speechclip")
    parser.add_argument("--checkpoint", default=None, help="SpeechCLIP .ckpt path; downloads from HF when omitted")
    parser.add_argument("--checkpoint-dir", default="competitors/data/checkpoints/speechclip")
    parser.add_argument("--image-cache", default="competitors/data/abo-images-speechclip")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    ckpt = args.checkpoint or download_checkpoint(args.checkpoint_dir)
    build_speechclip_abo_index(args.catalog, args.out, ckpt, args.image_cache, args.batch_size)
