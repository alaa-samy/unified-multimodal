"""FAISS index helpers mirroring the UniIR retrieval pipeline.

The official UniIR pipeline (src/common/mbeir_retriever.py) normalizes all
embeddings with faiss.normalize_L2 and indexes them into an IDMap InnerProduct
index, so cosine similarity is the raw score. We keep the same construction for
both competitor indexes and store the product-ID mapping in a sidecar JSON
because FAISS IDMap requires integer IDs while ABO product IDs are strings.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np


@dataclass
class AboIndex:
    index: Any
    product_ids: List[str]
    meta: Dict[str, Any]

    @property
    def dim(self) -> int:
        return self.index.d

    @property
    def size(self) -> int:
        return self.index.ntotal


def build_index(embeddings: np.ndarray, product_ids: List[str]) -> faiss.Index:
    if embeddings.shape[0] != len(product_ids):
        raise ValueError("embedding rows and product ids must match")
    vectors = np.ascontiguousarray(embeddings.astype("float32"))
    faiss.normalize_L2(vectors)
    index = faiss.IndexIDMap(faiss.IndexFlatIP(vectors.shape[1]))
    index.add_with_ids(vectors, np.arange(vectors.shape[0], dtype="int64"))
    return index


def save_index(index: faiss.Index, product_ids: List[str], meta: Dict[str, Any], out_dir: str) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path / "abo_images.faiss"))
    with open(out_path / "abo_ids.json", "w", encoding="utf-8") as f:
        json.dump(product_ids, f)
    payload = {"built_at": datetime.now(timezone.utc).isoformat(), "dim": index.d, "size": int(index.ntotal), **meta}
    with open(out_path / "index_meta.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def load_index(index_dir: str) -> AboIndex:
    index_path = Path(index_dir) / "abo_images.faiss"
    ids_path = Path(index_dir) / "abo_ids.json"
    meta_path = Path(index_dir) / "index_meta.json"
    if not index_path.exists() or not ids_path.exists():
        raise FileNotFoundError(f"ABO index not found under {index_dir}; build it first")
    index = faiss.read_index(str(index_path))
    with open(ids_path, "r", encoding="utf-8") as f:
        product_ids = json.load(f)
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return AboIndex(index=index, product_ids=product_ids, meta=meta)


def normalize_query(vector: np.ndarray) -> np.ndarray:
    query = np.ascontiguousarray(vector.astype("float32").reshape(1, -1))
    faiss.normalize_L2(query)
    return query


def search_index(index: faiss.Index, query_vector: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
    if index.ntotal == 0:
        return [], []
    distances, indices = index.search(normalize_query(query_vector), min(k, int(index.ntotal)))
    rows = [(int(i), float(s)) for i, s in zip(indices[0], distances[0]) if i != -1]
    return [r[0] for r in rows], [r[1] for r in rows]


def fuse_ranked_candidates(
    modality_results: List[Tuple[List[int], List[float]]], product_ids: List[str], top_k: int
) -> List[Dict[str, Any]]:
    """UniIR score-level fusion: sum the cosine scores a candidate earns per query modality."""
    fused: Dict[str, float] = {}
    for rows, scores in modality_results:
        for row, score in zip(rows, scores):
            product_id = product_ids[row]
            fused[product_id] = fused.get(product_id, 0.0) + score
    ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[: max(1, top_k)]
    return [{"product_id": product_id, "score": round(score, 4)} for product_id, score in ranked]
