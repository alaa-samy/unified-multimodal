"""UniIR /search bridge service.

Implements the exact multipart contract documented in the testbed adapter
(client/src/providers/uniir/README.md):

    POST /search  fields: query_txt, query_image, instruction, candidate_modality, top_k
    -> {"results": [{"product_id": "B07TMH6289", "score": 0.91}], "total_results": N}

Ranking is UniIR score-level fusion over the UniIR-specific ABO FAISS index:
score(candidate) = cos(instruction+text, ...) + cos(query_image, candidate image).
"""
from __future__ import annotations

import io
import os
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from competitors.common.abo_catalog import catalog_version, load_catalog
from competitors.common.defaults import DEFAULT_INSTRUCTION
from competitors.common.faiss_index import AboIndex, fuse_ranked_candidates, load_index, search_index

MAX_TOP_K = 50
FUSION_CANDIDATES = 50


def create_app(
    index_dir: Optional[str] = None,
    catalog_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> FastAPI:
    api = FastAPI(title="UniIR ABO Search Bridge")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state: Dict[str, Any] = {"model": None, "index": None, "catalog": None, "error": None}

    def _initialize() -> None:
        if state["index"] is not None:
            return
        resolved_index_dir = index_dir or os.environ.get("UNIIR_INDEX_DIR", "indexes/uniir")
        resolved_catalog = catalog_path or os.environ.get("ABO_CATALOG_PATH", "competitors/data/abo-catalog.json")
        resolved_ckpt = checkpoint_path or os.environ.get("UNIIR_CHECKPOINT_PATH")
        from competitors.uniir_bridge.model import UniIRCLIPSF

        state["catalog"] = load_catalog(resolved_catalog)
        state["index"] = load_index(resolved_index_dir)
        state["model"] = UniIRCLIPSF(checkpoint_path=resolved_ckpt)
        meta = state["index"].meta
        print(f"UniIR bridge ready: {state['index'].size} ABO vectors, dim {state['index'].dim}, meta={meta}")

    def _get_model() -> UniIRCLIPSF:
        if state["model"] is None:
            _initialize()
        return state["model"]

    def _get_index() -> AboIndex:
        if state["index"] is None:
            _initialize()
        return state["index"]

    @api.get("/")
    def root() -> Dict[str, Any]:
        return {"service": "uniir-abo-search", "status": "ok" if state["index"] is not None else "cold"}

    @api.get("/health")
    def health() -> Dict[str, Any]:
        if state["index"] is None and state["error"] is None:
            try:
                _initialize()
            except Exception as exc:
                state["error"] = str(exc)
        return {
            "status": "ok" if state["index"] is not None else "error",
            "provider": "uniir",
            "model": "UniIR CLIP-SF Large",
            "index_size": state["index"].size if state["index"] else 0,
            "catalog_version": catalog_version(state["catalog"]) if state["catalog"] else None,
            "error": state["error"],
        }

    @api.post("/search")
    async def search(
        query_txt: str = Form(""),
        query_image: Optional[UploadFile] = File(None),
        instruction: str = Form(DEFAULT_INSTRUCTION),
        candidate_modality: str = Form("image"),
        top_k: int = Form(20),
    ) -> JSONResponse:
        try:
            _get_model()
            abo = _get_index()
            top_k_value = max(1, min(int(top_k), MAX_TOP_K))

            image_bytes = await query_image.read() if query_image is not None else None
            has_text = bool((query_txt or "").strip())
            has_image = image_bytes is not None and len(image_bytes) > 0
            if not has_text and not has_image:
                return JSONResponse(status_code=400, content={"error": "query_txt or query_image is required"})

            model = state["model"]
            modality_results = []
            if has_text:
                text_vector = model.encode_text(query_txt, instruction)
                modality_results.append(search_index(abo.index, text_vector, FUSION_CANDIDATES))
            if has_image:
                image_vector = model.encode_image(Image.open(io.BytesIO(image_bytes)))
                modality_results.append(search_index(abo.index, image_vector, FUSION_CANDIDATES))

            results = fuse_ranked_candidates(modality_results, abo.product_ids, top_k_value)[:top_k_value]
            return JSONResponse(
                content={
                    "results": results,
                    "total_results": len(results),
                    "query": query_txt or "",
                    "provider": "uniir",
                    "catalog_version": catalog_version(state["catalog"]),
                    "modalities": ["text"] * has_text + ["image"] * has_image,
                    "candidate_modality": candidate_modality,
                }
            )
        except FileNotFoundError as exc:
            return JSONResponse(status_code=503, content={"error": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    return api
