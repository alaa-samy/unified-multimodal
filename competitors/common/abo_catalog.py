"""Shared ABO catalog loader for the competitor bridges.

The candidate pool is the exact 75-product ABO fixture the testbed storefront
exposes (exported from shared/catalog.ts to abo-catalog.json), so every FAISS
row maps back to a stable product ID the UI can render.
"""
from __future__ import annotations

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "abo-catalog.json"


def load_catalog(path: Optional[str] = None) -> Dict[str, Any]:
    catalog_path = Path(path) if path else DEFAULT_CATALOG_PATH
    with open(catalog_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("products"), list):
        raise ValueError(f"{catalog_path} must contain a 'products' list")
    return data


def catalog_version(catalog: Dict[str, Any]) -> str:
    return str(catalog.get("version") or catalog.get("catalogVersion") or "abo-unknown")


def image_url(product: Dict[str, Any]) -> str:
    return str(product.get("image") or product.get("image_url") or "")


def download_images(products: List[Dict[str, Any]], out_dir: str, timeout: int = 30) -> Dict[str, str]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    def fetch(product: Dict[str, Any]) -> None:
        url = image_url(product)
        if not url:
            return
        target = out_path / f"{product['id']}.jpg"
        if target.exists() and target.stat().st_size > 0:
            paths[product["id"]] = str(target)
            return
        request = urllib.request.Request(url, headers={"User-Agent": "abo-index-builder/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                target.write_bytes(response.read())
            paths[product["id"]] = str(target)
        except Exception as exc:
            print(f"Failed to download {url}: {exc}")

    with ThreadPoolExecutor(max_workers=min(16, len(products) or 1)) as pool:
        list(pool.map(fetch, products))

    return paths
