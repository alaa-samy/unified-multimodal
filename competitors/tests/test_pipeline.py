"""Runnable pipeline tests for the competitor bridges.

These run without torch or the model checkpoints: a deterministic mock encoder
replaces the neural encoders while FAISS, the ABO catalog loader, the fusion
ranking, and the metric code are exercised for real.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from competitors.common.abo_catalog import catalog_version, load_catalog
from competitors.common.faiss_index import build_index, fuse_ranked_candidates, load_index, save_index, search_index
from competitors.common.ranking_metrics import aggregate, mrr, recall_at_k


class MockEncoder:
    """Deterministic stand-in for a CLIP-style dual encoder.

    Vectors are built from a fixed hash of the input so the same text/image
    always yields the same direction, and related keywords share a base
    direction, which makes ranking assertions stable.
    """

    dim = 64

    def __init__(self) -> None:
        self.rng = np.random.default_rng(2024)

    def _vector(self, seed_text: str) -> np.ndarray:
        seed = abs(hash(seed_text)) % (2**32)
        rng = np.random.default_rng(seed)
        vector = rng.standard_normal(self.dim).astype("float32")
        return vector / np.linalg.norm(vector)

    def encode(self, key: str) -> np.ndarray:
        return self._vector(key)


@pytest.fixture(scope="module")
def abo_catalog():
    catalog = load_catalog()
    assert len(catalog["products"]) > 0
    return catalog


@pytest.fixture(scope="module")
def indexed_products(abo_catalog, tmp_path_factory):
    products = abo_catalog["products"]
    encoder = MockEncoder()
    embeddings = np.stack([encoder.encode(p["id"]) for p in products]).astype("float32")
    out_dir = tmp_path_factory.mktemp("index")
    index = build_index(embeddings, [p["id"] for p in products])
    save_index(
        index,
        [p["id"] for p in products],
        {"provider": "mock", "catalog_version": catalog_version(abo_catalog)},
        str(out_dir),
    )
    return load_index(str(out_dir)), encoder


def test_index_roundtrip_and_sizes(indexed_products, abo_catalog):
    abo, _ = indexed_products
    assert abo.size == len(abo_catalog["products"])
    assert abo.product_ids == [p["id"] for p in abo_catalog["products"]]
    assert abo.dim == MockEncoder.dim
    assert abo.index.ntotal == abo.size


def test_self_retrieval_ranks_the_product_first(indexed_products):
    abo, encoder = indexed_products
    first_id = abo.product_ids[0]
    rows, scores = search_index(abo.index, encoder.encode(first_id), 5)
    ranked = [abo.product_ids[row] for row in rows]
    assert ranked[0] == first_id
    assert scores[0] == pytest.approx(1.0, abs=1e-4)
    assert scores == sorted(scores, reverse=True)


def test_search_respects_k_and_drops_invalid_rows(indexed_products):
    abo, encoder = indexed_products
    rows, scores = search_index(abo.index, encoder.encode(abo.product_ids[1]), min(3, abo.size))
    assert len(rows) == len(scores) <= 3


def test_fusion_sums_modality_scores_and_sorts(indexed_products):
    abo, encoder = indexed_products
    target = abo.product_ids[0]
    text_rows, text_scores = search_index(abo.index, encoder.encode(target), 10)
    image_rows, image_scores = search_index(abo.index, encoder.encode(target), 10)
    fused = fuse_ranked_candidates([(text_rows, text_scores), (image_rows, image_scores)], abo.product_ids, 5)
    assert fused[0]["product_id"] == target
    assert len(fused) <= 5
    scores = [entry["score"] for entry in fused]
    assert scores == sorted(scores, reverse=True)


def test_fusion_requires_candidate_overlap(indexed_products):
    abo, _ = indexed_products
    fused = fuse_ranked_candidates([([0], [0.9]), ([1], [0.8])], abo.product_ids, 5)
    assert [entry["product_id"] for entry in fused] == [abo.product_ids[0], abo.product_ids[1]]


def test_metrics_match_uniir_hit_rate_definition():
    assert recall_at_k(["a", "b", "c"], ["c"], 3) == 1.0
    assert recall_at_k(["a", "b", "c"], ["c"], 2) == 0.0
    assert mrr(["a", "b", "c"], ["b"]) == 0.5
    assert mrr(["a", "b", "c"], ["missing"]) == 0.0


def test_aggregate_reports_all_modalities():
    rows = [
        {"recall@1": 1.0, "recall@5": 1.0, "recall@10": 1.0, "mrr": 1.0, "latency_ms": 20.0},
        {"recall@1": 0.0, "recall@5": 1.0, "recall@10": 1.0, "mrr": 0.5, "latency_ms": 40.0},
    ]
    summary = aggregate(rows)
    assert summary["recall@1"] == 0.5
    assert summary["recall@5"] == 1.0
    assert summary["mrr"] == 0.75
    assert summary["mean_latency_ms"] == 30.0
    assert summary["queries"] == 2.0


def test_empty_aggregate_is_safe():
    summary = aggregate([])
    assert summary["recall@1"] == 0.0
    assert summary["mrr"] == 0.0
    assert summary["queries"] == 0.0


def test_abo_catalog_has_stable_ids_and_images(abo_catalog):
    ids = [p["id"] for p in abo_catalog["products"]]
    assert len(ids) == len(set(ids))
    assert all(p.get("image") for p in abo_catalog["products"])
    assert all(len(p["id"]) == 10 for p in abo_catalog["products"])


def test_index_meta_records_provider(tmp_path, abo_catalog):
    import faiss

    index = build_index(np.eye(4, dtype="float32"), ["a", "b", "c", "d"])
    out = save_index(index, ["a", "b", "c", "d"], {"provider": "uniir"}, str(tmp_path))
    meta = json.loads((out / "index_meta.json").read_text())
    assert meta["provider"] == "uniir"
    assert meta["size"] == 4
    assert faiss.read_index(str(out / "abo_images.faiss")).ntotal == 4
