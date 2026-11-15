"""Build the UniIR-specific ABO FAISS index.

Embeds every ABO candidate image with the fine-tuned UniIR CLIP-SF encoder and
writes indexes/uniir/ (abo_images.faiss, abo_ids.json, index_meta.json).

Local run (needs the checkpoint and GPU):
    python -m competitors.uniir_bridge.build_index --checkpoint /path/clip_sf_large.pth

On Modal (GPU):
    modal run competitors/uniir_bridge/modal_app.py::build-index
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from competitors.common.abo_catalog import catalog_version, download_images, image_url, load_catalog
from competitors.common.faiss_index import build_index, save_index
from competitors.uniir_bridge.model import UniIRCLIPSF, download_checkpoint


def build_uniir_abo_index(
    catalog_path: str,
    out_dir: str,
    checkpoint_path: str,
    image_cache_dir: str = "competitors/data/abo-images-uniir",
    batch_size: int = 32,
) -> Dict[str, object]:
    catalog = load_catalog(catalog_path)
    products = catalog["products"]
    with_images = [p for p in products if image_url(p)]
    print(f"Embedding {len(with_images)} ABO images with UniIR CLIP-SF ({len(products)} in catalog)")

    image_paths = download_images(with_images, image_cache_dir)
    model = UniIRCLIPSF(checkpoint_path=checkpoint_path)

    ready = [(p, image_paths[p["id"]]) for p in with_images if p["id"] in image_paths]
    embeddings = model.encode_image_batch([Image.open(path) for _, path in ready], batch_size=batch_size)
    product_ids: List[str] = [p["id"] for p, _ in ready]

    index = build_index(embeddings, product_ids)
    out_path = save_index(
        index,
        product_ids,
        {
            "provider": "uniir",
            "model": f"UniIR CLIP-SF Large ({model.clip_model_name})",
            "checkpoint": str(checkpoint_path),
            "catalog_version": catalog_version(catalog),
            "modality": "image candidates, instruction-guided text/image queries",
        },
        out_dir,
    )
    skipped = len(products) - len(ready)
    print(f"UniIR ABO index: {index.ntotal} vectors, dim {index.d}, saved to {out_path} (skipped {skipped} without images)")
    return {"size": int(index.ntotal), "dim": int(index.d), "path": str(out_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="competitors/data/abo-catalog.json")
    parser.add_argument("--out", default="indexes/uniir")
    parser.add_argument("--checkpoint", default=None, help="clip_sf_large.pth path; downloads from HF when omitted")
    parser.add_argument("--checkpoint-dir", default="competitors/data/checkpoints/uniir")
    parser.add_argument("--image-cache", default="competitors/data/abo-images-uniir")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    ckpt = args.checkpoint or download_checkpoint(args.checkpoint_dir)
    build_uniir_abo_index(args.catalog, args.out, ckpt, args.image_cache, args.batch_size)
