#!/usr/bin/env python3
"""Recompute robust per-product color histograms for the Pix3D catalog.

Run from backend/ after the Pix3D images and clean_products.json exist:
    python recompute_metadata.py

The search service uses these histograms as a soft color signal. This script
updates only ``color_hist`` in clean_products.json and never changes IDs,
image paths, embeddings, or product ordering.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

COLOR_NAMES = ("black", "white", "gray", "brown", "beige", "red", "orange", "yellow", "green", "blue", "navy", "purple", "pink")
# Representative RGB colors. These are intentionally broad visual buckets,
# because the search layer gives partial credit to related colors.
COLOR_RGB = np.array([
    (28, 28, 30), (242, 242, 238), (128, 130, 132), (112, 72, 42),
    (205, 184, 151), (180, 42, 42), (215, 104, 42), (218, 180, 42),
    (75, 135, 72), (55, 105, 185), (35, 55, 115), (125, 75, 155), (200, 90, 150),
], dtype=np.float32)


def _canonical_color(value: object) -> str | None:
    text = str(value or "").strip().lower()
    aliases = {
        "grey": "gray", "silver": "gray", "charcoal": "gray",
        "tan": "beige", "cream": "beige", "ivory": "beige",
        "burgundy": "red", "maroon": "red", "crimson": "red",
        "violet": "purple", "rose": "pink", "bleu": "blue",
    }
    text = aliases.get(text, text)
    return text if text in COLOR_NAMES else None


def color_histogram(image_path: Path, prior_color: object = "") -> dict[str, float]:
    """Estimate a normalized color distribution while reducing background bias.

    A center crop is used because Pix3D product photos commonly include a
    large neutral background. Pixels with very low saturation are retained so
    black/white/gray furniture remains searchable, but the outer image border
    is excluded from the estimate.
    """
    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            width, height = image.size
            left, top = int(width * 0.12), int(height * 0.12)
            right, bottom = int(width * 0.88), int(height * 0.88)
            image = image.crop((left, top, max(left + 1, right), max(top + 1, bottom)))
            image.thumbnail((96, 96))
            pixels = np.asarray(image, dtype=np.float32).reshape(-1, 3)
    except Exception:
        fallback = _canonical_color(prior_color)
        return {fallback: 1.0} if fallback else {}

    if len(pixels) == 0:
        fallback = _canonical_color(prior_color)
        return {fallback: 1.0} if fallback else {}

    # Downweight near-white/near-gray pixels when they dominate the whole
    # image, which is usually the studio background rather than the product.
    brightness = pixels.mean(axis=1)
    spread = pixels.max(axis=1) - pixels.min(axis=1)
    keep = np.ones(len(pixels), dtype=bool)
    background_like = (brightness > 238) & (spread < 18)
    if background_like.mean() > 0.45:
        keep = ~background_like
    selected = pixels[keep] if keep.any() else pixels

    # Neutral pixels (low chroma) are split by brightness: black / gray / white.
    # Chromatic pixels are matched on intensity-normalised RGB so shading does not
    # change the hue bucket. (The old code normalised EVERY pixel by its mean
    # intensity, which erases brightness: black, gray and white all became ~(128,128,128)
    # and a black sofa was labelled "white".)
    idx = {name: i for i, name in enumerate(COLOR_NAMES)}
    neutral = (selected.max(axis=1) - selected.min(axis=1)) < 28
    labels = np.empty(len(selected), dtype=np.int64)
    value = selected.mean(axis=1)
    labels[neutral] = np.where(value[neutral] < 75, idx["black"],
                               np.where(value[neutral] > 195, idx["white"], idx["gray"]))
    chroma = ~neutral
    if chroma.any():
        chroma_ids = [i for i, n in enumerate(COLOR_NAMES) if n not in ("black", "white", "gray")]
        pix = selected[chroma]
        normalized = pix / np.maximum(pix.mean(axis=1, keepdims=True), 1.0) * 128.0
        refs = COLOR_RGB[chroma_ids] / np.maximum(COLOR_RGB[chroma_ids].mean(axis=1, keepdims=True), 1.0) * 128.0
        nearest = np.argmin(((normalized[:, None, :] - refs[None, :, :]) ** 2).mean(axis=2), axis=1)
        labels[chroma] = np.asarray(chroma_ids)[nearest]
    counts = np.bincount(labels, minlength=len(COLOR_NAMES)).astype(np.float32)

    # Keep meaningful colors only, but preserve the strongest bucket.
    counts[counts < max(2.0, counts.sum() * 0.025)] = 0
    if counts.sum() == 0:
        counts[np.argmax(np.bincount(labels, minlength=len(COLOR_NAMES)))] = 1
    counts /= counts.sum()
    return {COLOR_NAMES[i]: round(float(value), 4) for i, value in enumerate(counts) if value > 0}


def recompute(products_path: Path, images_dir: Path) -> tuple[int, int]:
    products = json.loads(products_path.read_text(encoding="utf-8"))
    updated = 0
    missing = 0
    for product in products:
        relative = product.get("image_path") or ""
        image_path = images_dir / relative
        if not image_path.exists():
            missing += 1
            continue
        hist = color_histogram(image_path, product.get("baseColour") or product.get("color"))
        if hist:
            product["color_hist"] = hist
            updated += 1

    fd, temporary = tempfile.mkstemp(prefix="clean_products.", suffix=".json", dir=str(products_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(products, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, products_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return updated, missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path, default=Path("data/clean_products.json"))
    parser.add_argument("--images", type=Path, default=Path("data/images"))
    args = parser.parse_args()
    if not args.products.exists():
        raise SystemExit(f"Products file not found: {args.products}")
    updated, missing = recompute(args.products, args.images)
    print(f"[METADATA] Updated color histograms for {updated} products")
    if missing:
        print(f"[METADATA] Skipped {missing} products with missing images")


if __name__ == "__main__":
    main()