"""Stable contracts for dataset and search-provider comparisons.

The production search route can keep using the local implementation while
alternative providers are added behind this boundary. A provider must return
ranked product IDs and timing metadata; conversion experiments must use the
same query set and catalog labels for every provider.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Iterable, Protocol, Sequence


@dataclass(frozen=True)
class SearchQuery:
    query_id: str
    text: str = ""
    image_path: str | None = None
    audio_path: str | None = None
    relevant_product_ids: frozenset[str] = frozenset()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchHit:
    product_id: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetPlugin(Protocol):
    name: str

    def products(self) -> Iterable[dict[str, Any]]: ...

    def queries(self) -> Iterable[SearchQuery]: ...


class SearchProvider(Protocol):
    name: str

    def search(self, query: SearchQuery, top_k: int = 10) -> Sequence[SearchHit]: ...


@dataclass(frozen=True)
class BenchmarkResult:
    provider: str
    dataset: str
    query_count: int
    recall_at_5: float
    recall_at_10: float
    mrr: float
    mean_latency_ms: float


def evaluate_provider(provider: SearchProvider, dataset: DatasetPlugin, top_k: int = 10) -> BenchmarkResult:
    """Evaluate a provider on fixed relevance labels, not simulated purchases."""
    queries = list(dataset.queries())
    if not queries:
        raise ValueError("Dataset plugin returned no benchmark queries")

    hits_5 = hits_10 = reciprocal_rank = 0.0
    latencies: list[float] = []
    for query in queries:
        started = perf_counter()
        hits = list(provider.search(query, top_k=top_k))
        latencies.append((perf_counter() - started) * 1000)
        relevant = query.relevant_product_ids
        ids = [hit.product_id for hit in hits]
        hits_5 += float(bool(relevant.intersection(ids[:5])))
        hits_10 += float(bool(relevant.intersection(ids[:10])))
        reciprocal_rank += next(
            (1.0 / (idx + 1) for idx, product_id in enumerate(ids) if product_id in relevant),
            0.0,
        )

    n = len(queries)
    return BenchmarkResult(
        provider=provider.name,
        dataset=dataset.name,
        query_count=n,
        recall_at_5=hits_5 / n,
        recall_at_10=hits_10 / n,
        mrr=reciprocal_rank / n,
        mean_latency_ms=sum(latencies) / n,
    )
