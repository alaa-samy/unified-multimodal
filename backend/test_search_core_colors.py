"""Unit tests for search_core.py color histogram binning and confidence.

Needs only numpy (no torch / FastAPI), so it runs anywhere:

    pytest backend/test_search_core_colors.py
"""
import numpy as np

from search_core import COLOR_NAMES, CatalogIndex, _CIDX


def _make_matrix(rows, dim=8):
    rng = np.random.default_rng(rows)
    return rng.standard_normal((rows, dim)).astype("float32")


def test_burgundy_basecolour_counts_as_red():
    """Point-2 fix: a 'Burgundy' baseColour label must land in the red family
    bin instead of being dropped (which left the product with an all-zero
    color histogram and zero credit for 'red ...' queries)."""
    products = [
        {"subCategory": "sofa", "baseColour": "Burgundy"},   # fallback label path
        {"subCategory": "sofa", "baseColour": "black"},
    ]
    idx = CatalogIndex(_make_matrix(2), products)

    red_scores = idx.color_scores("red")
    assert red_scores[0] == 1.0, "burgundy item must get full red credit"
    assert red_scores[1] == 0.0

    # and it is not silently lost in the histogram either
    assert idx.H[0, _CIDX["red"]] == 1.0
    assert idx.H.sum(axis=1)[0] == 1.0


def test_color_hist_keys_folded_into_families():
    """Alias keys inside a pre-computed color_hist ('burgundy', 'noir',
    'charcoal', 'cream') are folded into canonical families instead of being
    skipped."""
    products = [{
        "subCategory": "sofa",
        "color_hist": {"burgundy": 0.6, "noir": 0.4},
    }]
    idx = CatalogIndex(_make_matrix(1), products)
    assert idx.H[0, _CIDX["red"]] == 0.6
    assert idx.H[0, _CIDX["black"]] == 0.4
    assert abs(idx.H.sum() - 1.0) < 1e-6


def test_query_alias_burgundy_maps_to_red():
    from search_core import extract_color
    assert extract_color("burgundy leather sofa") == "red"


def test_black_sofa_results_are_not_low_confidence():
    """'search for black sofa' returned 10 black sofas while the UI apologised
    for a weak match, because raw CLIP text cosines on this catalog sit below
    LOW_CONF_REL. When the requested color is confirmed by the products' own
    histograms, the result is NOT low-confidence."""
    # All cosines < 0.229 => relevance < LOW_CONF_REL (0.35)
    vec = np.array([0.20, 0.18, 0.15, 0.10, 0.05, 0.0, 0.0, 0.0], dtype="float32")
    products = [{"subCategory": "sofa", "baseColour": "black"} for _ in range(5)]
    idx = CatalogIndex(np.eye(5, 8, dtype="float32"), products)

    hits, meta = idx.search(text_vec=vec, color="black", category="sofa", top_k=5)
    assert len(hits) == 5
    assert max(h["relevance"] for h in hits) < 0.35          # genuinely weak raw cosine
    assert meta["low_confidence"] is False, (
        "results whose own histograms confirm the requested color must not "
        "be flagged as a weak match"
    )


def test_weak_results_without_color_still_low_confidence():
    """No color requested and low relevance -> low_confidence stays True."""
    vec = np.array([0.20, 0.18, 0.15, 0.10, 0.05, 0.0, 0.0, 0.0], dtype="float32")
    products = [{"subCategory": "sofa", "baseColour": "black"} for _ in range(5)]
    idx = CatalogIndex(np.eye(5, 8, dtype="float32"), products)
    hits, meta = idx.search(text_vec=vec, category="sofa", top_k=5)
    assert meta["low_confidence"] is True


def test_color_mismatch_still_low_confidence():
    """Color requested but results don't actually have it -> stays honest."""
    vec = np.array([0.20, 0.18, 0.15, 0.10, 0.05, 0.0, 0.0, 0.0], dtype="float32")
    products = [{"subCategory": "sofa", "baseColour": "black"} for _ in range(5)]
    idx = CatalogIndex(np.eye(5, 8, dtype="float32"), products)
    hits, meta = idx.search(text_vec=vec, color="red", category="sofa", top_k=5)
    assert max(h["color_match"] for h in hits) < 0.5
    assert meta["low_confidence"] is True


def test_all_color_names_have_families():
    assert set(COLOR_NAMES) == set(_CIDX)
