"""Modality-level ranking metrics, aligned with the UniIR recall definition.

UniIR's compute_recall_at_k returns 1.0 when any relevant candidate appears in
the top-k, so recall@k here is hit-rate@k. MRR and latency complement it.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return 1.0 if relevant.intersection(ranked_ids[:k]) else 0.0


def mrr(ranked_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    relevant = set(relevant_ids)
    for rank, product_id in enumerate(ranked_ids, start=1):
        if product_id in relevant:
            return 1.0 / rank
    return 0.0


def aggregate(query_results: List[Dict[str, object]], ks=(1, 5, 10)) -> Dict[str, float]:
    if not query_results:
        return {f"recall@{k}": 0.0 for k in ks} | {"mrr": 0.0, "mean_latency_ms": 0.0, "queries": 0.0}
    summary: Dict[str, float] = {}
    for k in ks:
        summary[f"recall@{k}"] = sum(float(r.get(f"recall@{k}", 0.0)) for r in query_results) / len(query_results)
    summary["mrr"] = sum(float(r.get("mrr", 0.0)) for r in query_results) / len(query_results)
    summary["mean_latency_ms"] = sum(float(r.get("latency_ms", 0.0)) for r in query_results) / len(query_results)
    summary["queries"] = float(len(query_results))
    return summary
