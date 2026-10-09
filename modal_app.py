import io
import json
import csv
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image as PILImage
from transformers import CLIPModel, CLIPProcessor

import modal

app = modal.App("furniture-search-api")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi==0.104.1",
        "python-multipart==0.0.6",
        "Pillow==10.1.0",
        "numpy==1.24.3",
        "torch==2.1.0",
        "transformers==4.35.2",
        "faiss-cpu==1.7.4",
    )
)

volume = modal.Volume.from_name("furniture-data-cache", create_if_missing=True)

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR = "/cache/data"


@app.function(
    image=image,
    volumes={"/cache": volume},
    memory=2048,
    timeout=600,
    min_containers=0,
    max_containers=3,
    scaledown_window=60,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def serve():
    api = FastAPI(title="Furniture Search API")

    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _clip_model = None
    _clip_processor = None
    _data_loaded = False
    _products: List[Dict[str, Any]] = []
    _data_source = "none"
    _device = "cpu"

    def _get_device() -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _build_demo_catalog() -> List[Dict[str, Any]]:
        return [
            {
                "id": "demo-1",
                "name": "Modern beige sofa",
                "description": "Comfortable beige sofa for living room furniture with soft cushions",
                "category": "sofa",
                "image": None,
                "has_ar": False,
                "ar_type": None,
                "ar_model_url": None,
            },
            {
                "id": "demo-2",
                "name": "Wooden dining table",
                "description": "Solid wood dining table for family meals and modern interiors",
                "category": "table",
                "image": None,
                "has_ar": False,
                "ar_type": None,
                "ar_model_url": None,
            },
        ]

    def _normalize_text(value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split())

    def _build_product_text(product: Dict[str, Any]) -> str:
        parts = [
            product.get("name", ""),
            product.get("description", ""),
            product.get("category", ""),
            product.get("brand", ""),
            product.get("color", ""),
            product.get("material", ""),
        ]
        return " ".join([_normalize_text(p) for p in parts if _normalize_text(p)])

    def _safe_read_json(path: str) -> Optional[List[Dict[str, Any]]]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [dict(x) for x in data if isinstance(x, dict)]
            if isinstance(data, dict):
                if isinstance(data.get("products"), list):
                    return [dict(x) for x in data["products"] if isinstance(x, dict)]
                if isinstance(data.get("items"), list):
                    return [dict(x) for x in data["items"] if isinstance(x, dict)]
            return None
        except Exception:
            return None

    def _safe_read_csv(path: str) -> Optional[List[Dict[str, Any]]]:
        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            return [dict(r) for r in rows]
        except Exception:
            return None

    def _resolve_local_path(path_value: Optional[str]) -> Optional[str]:
        if not path_value:
            return None

        raw = str(path_value).strip()
        if not raw:
            return None

        candidates = [
            raw,
            os.path.join(CACHE_DIR, raw),
            os.path.join(CACHE_DIR, "data", raw),
            str(PROJECT_ROOT / raw),
            str(PROJECT_ROOT / "backend" / "data" / raw),
        ]

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        return None

    def _model_key(path_value: Optional[str]) -> Optional[str]:
        if not path_value:
            return None

        p = str(path_value).replace("\\", "/").rstrip("/")
        if p.endswith("/model.obj") or p.endswith("/model.glb") or p.endswith("/model"):
            return os.path.basename(os.path.dirname(p)).lower()

        if p.endswith(".obj") or p.endswith(".glb"):
            p = p[:-4]

        return os.path.basename(p).lower()

    def _load_pix3d_index() -> Dict[str, Dict[str, Any]]:
        possible_paths = [
            str(PROJECT_ROOT / "backend" / "data" / "pix3d" / "pix3d.json"),
            str(PROJECT_ROOT / "backend" / "data" / "pix3d.json"),
            os.path.join(CACHE_DIR, "pix3d", "pix3d.json"),
            os.path.join(CACHE_DIR, "data", "pix3d", "pix3d.json"),
            os.path.join(CACHE_DIR, "pix3d.json"),
            os.path.join(CACHE_DIR, "data", "pix3d.json"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        index = {}
                        for item in data:
                            if isinstance(item, dict):
                                key = _model_key(item.get("model"))
                                if key:
                                    index[key] = item
                        return index
                except Exception:
                    continue

        return {}

    def _load_clean_products_catalog() -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        possible_paths = [
            str(PROJECT_ROOT / "backend" / "data" / "clean_products.json"),
            os.path.join(CACHE_DIR, "clean_products.json"),
            os.path.join(CACHE_DIR, "data", "clean_products.json"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                data = _safe_read_json(path)
                if data:
                    return data, path

        return None, None

    def _find_catalog_files() -> List[str]:
        candidates = []

        for path in [
            str(PROJECT_ROOT / "backend" / "data" / "clean_products.json"),
            str(PROJECT_ROOT / "backend" / "data" / "products.json"),
            str(PROJECT_ROOT / "backend" / "data" / "catalog.json"),
            "/cache/data/catalog.json",
            "/cache/data/products.json",
            "/cache/data/data/catalog.json",
            "/cache/data/data/products.json",
            os.path.join(CACHE_DIR, "catalog.json"),
            os.path.join(CACHE_DIR, "products.json"),
            os.path.join(CACHE_DIR, "data", "catalog.json"),
            os.path.join(CACHE_DIR, "data", "products.json"),
        ]:
            if os.path.exists(path):
                candidates.append(path)

        if candidates:
            return candidates

        for root, _, files in os.walk(CACHE_DIR):
            for name in files:
                if name.lower().endswith((".json", ".csv")):
                    candidates.append(os.path.join(root, name))
        return candidates

    def _load_catalog() -> tuple[List[Dict[str, Any]], str]:
        cleaned, cleaned_path = _load_clean_products_catalog()
        if cleaned:
            pix3d_index = _load_pix3d_index()

            normalized = []
            for idx, product in enumerate(cleaned):
                if not isinstance(product, dict):
                    continue

                name = (
                    product.get("productDisplayName")
                    or product.get("name")
                    or product.get("title")
                    or f"Product {idx}"
                )
                category = (
                    product.get("subCategory")
                    or product.get("category")
                    or product.get("masterCategory")
                    or "Furniture"
                )
                description_parts = [
                    product.get("masterCategory"),
                    product.get("subCategory"),
                    product.get("baseColour"),
                    product.get("description"),
                ]
                description = " ".join(
                    [_normalize_text(p) for p in description_parts if _normalize_text(p)]
                )

                image_path = product.get("image_path") or product.get("image") or product.get("image_url")
                resolved_image = _resolve_local_path(image_path)

                has_ar = bool(product.get("has_3d_model"))
                ar_model = product.get("glb_path") or product.get("ar_model_url") or product.get("ar_model_path")

                ar_payload = {
                    "has_ar": has_ar,
                    "ar_type": "model" if has_ar else None,
                    "ar_model_url": ar_model,
                }

                pix3d_match = None
                if ar_model:
                    pix3d_match = pix3d_index.get(_model_key(ar_model), {})

                normalized.append(
                    {
                        "id": product.get("id", idx),
                        "name": name,
                        "description": description,
                        "category": category,
                        "brand": product.get("brand", ""),
                        "color": product.get("baseColour") or product.get("color", ""),
                        "material": product.get("material", ""),
                        "image": resolved_image,
                        "image_path": resolved_image,
                        "image_url": image_path,
                        "has_ar": ar_payload["has_ar"],
                        "ar_type": ar_payload["ar_type"],
                        "ar_model_url": ar_payload["ar_model_url"],
                        "pix3d_image": pix3d_match.get("img") if isinstance(pix3d_match, dict) else None,
                        "pix3d_category": pix3d_match.get("category") if isinstance(pix3d_match, dict) else None,
                    }
                )

            return normalized, cleaned_path

        catalog_files = _find_catalog_files()
        for path in catalog_files:
            if path.lower().endswith(".json"):
                data = _safe_read_json(path)
                if data:
                    return data, path
            elif path.lower().endswith(".csv"):
                data = _safe_read_csv(path)
                if data:
                    return data, path

        return _build_demo_catalog(), "demo-catalog"

    def initialize_clip():
        nonlocal _clip_model, _clip_processor, _device
        if _clip_model is not None and _clip_processor is not None:
            return
        _device = _get_device()
        print(f"Loading CLIP on {_device}...")
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _clip_model.to(_device)
        _clip_model.eval()

    def initialize_data():
        nonlocal _products, _data_loaded, _data_source
        if _data_loaded:
            return

        print("Initializing product catalog...")
        products, source = _load_catalog()

        initialize_clip()

        for product in products:
            product["text"] = _build_product_text(product)
            product["text_embedding"] = _embed_text(product["text"])

            image_path = product.get("image") or product.get("image_path") or product.get("image_url")
            if image_path and os.path.exists(str(image_path)):
                product["image_embedding"] = _embed_image_from_path(str(image_path))
            else:
                product["image_embedding"] = None

        _products = products
        _data_source = source
        _data_loaded = True
        print(f"Catalog loaded: {len(_products)} products from {_data_source}")

    def _embed_text(text: str):
        nonlocal _clip_model, _clip_processor
        if _clip_model is None or _clip_processor is None:
            initialize_clip()

        inputs = _clip_processor(text=[text], return_tensors="pt", padding=True)
        with torch.inference_mode():
            embedding = _clip_model.get_text_features(**inputs)
        embedding = embedding[0].cpu().numpy().astype(np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm > 1e-12:
            embedding = embedding / norm
        return embedding

    def _embed_image_bytes(image_bytes: bytes):
        nonlocal _clip_model, _clip_processor
        if _clip_model is None or _clip_processor is None:
            initialize_clip()

        try:
            image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
            inputs = _clip_processor(images=image, return_tensors="pt")
            with torch.inference_mode():
                embedding = _clip_model.get_image_features(**inputs)
            embedding = embedding[0].cpu().numpy().astype(np.float32)
            norm = float(np.linalg.norm(embedding))
            if norm > 1e-12:
                embedding = embedding / norm
            return embedding
        except Exception:
            return None

    def _embed_image_from_path(path: str):
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                return _embed_image_bytes(f.read())
        except Exception:
            return None

    def _cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        if a is None or b is None:
            return 0.0
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def search_products(query: str, image_bytes: Optional[bytes] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        initialize_data()
        if not _products:
            return []

        query = (query or "").strip() or "furniture product"
        query_text_embedding = _embed_text(query)
        query_image_embedding = _embed_image_bytes(image_bytes) if image_bytes else None

        scored = []
        for product in _products:
            text_score = _cosine_similarity(query_text_embedding, product.get("text_embedding"))
            image_score = 0.0
            if query_image_embedding is not None and product.get("image_embedding") is not None:
                image_score = _cosine_similarity(query_image_embedding, product.get("image_embedding"))

            if query_image_embedding is not None:
                final_score = 0.7 * text_score + 0.3 * image_score
            else:
                final_score = text_score

            scored.append(
                {
                    "id": product.get("id", ""),
                    "name": product.get("name", ""),
                    "description": product.get("description", ""),
                    "category": product.get("category", ""),
                    "score": round(float(final_score), 4),
                    "has_ar": bool(product.get("has_ar")),
                    "ar_type": product.get("ar_type"),
                    "ar_model_url": product.get("ar_model_url"),
                    "pix3d_image": product.get("pix3d_image"),
                    "pix3d_category": product.get("pix3d_category"),
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[: max(1, min(int(top_k), 20))]

    @api.get("/")
    def root():
        return {"message": "Furniture search API is running"}

    @api.get("/health")
    def health():
        initialize_data()
        return {
            "status": "ok",
            "catalog_size": len(_products),
            "data_source": _data_source,
        }

    @api.get("/wake-up")
    def wake_up():
        initialize_data()
        return {"status": "waking-up"}

    @api.post("/search/text")
    def search_text(query: str = Form(...), top_k: int = Form(10)):
        try:
            top_k_value = max(1, min(int(top_k), 20))
            results = search_products(query=query, top_k=top_k_value)
            return JSONResponse(content={"query": query, "results": results, "data_source": _data_source})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    @api.post("/search/intelligent")
    async def intelligent_search(
        query: str = Form(""),
        image: Optional[UploadFile] = File(None),
        top_k: int = Form(10),
    ):
        try:
            image_bytes = None
            if image is not None:
                image_bytes = await image.read()

            top_k_value = max(1, min(int(top_k), 20))
            results = search_products(query=query, image_bytes=image_bytes, top_k=top_k_value)
            return JSONResponse(
                content={
                    "query": query,
                    "results": results,
                    "data_source": _data_source,
                    "image_received": image is not None,
                }
            )
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    return api