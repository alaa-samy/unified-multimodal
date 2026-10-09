"""
search_core.py  -  pure search logic (no FastAPI / torch imports, easy to test)

What changed vs. the old main.py
  1. Text queries are matched against the IMAGE embeddings (CLIP text->image),
     not against the useless "Brown Chair" text index.
  2. Text + image (+ voice, which is just text) are fused at SCORE level with
     per-modality z-normalisation, instead of a plain mean of two vectors that
     live in different regions of CLIP space.
  3. Color is a soft boost from a per-product color histogram (computed from the
     object mask, see recompute_metadata.py), not a hard filter on one bad label.
  4. Category words (sofa/couch/wardrobe/closet...) become a hard filter, because
     Pix3D labels are reliable. For image queries the category is inferred from
     the nearest neighbours and applied as a soft boost.
  5. Whole-word color matching (the old code matched "tan" inside "standing").
"""
import re
import difflib
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

# ------------------------------------------------------------------ colors
COLOR_NAMES = ["black", "white", "gray", "brown", "beige", "red", "orange",
               "yellow", "green", "blue", "navy", "purple", "pink"]
_CIDX = {c: i for i, c in enumerate(COLOR_NAMES)}

COLOR_ALIASES = {
    "black": ["black", "noir"], "white": ["white", "blanc"],
    "gray": ["gray", "grey", "silver", "charcoal", "gris"],
    "brown": ["brown", "marron", "chocolate"],
    "beige": ["beige", "tan", "cream", "ivory"],
    "red": ["red", "burgundy", "maroon", "crimson", "rouge"],
    "orange": ["orange"], "yellow": ["yellow", "jaune"],
    "green": ["green", "vert"], "blue": ["blue", "bleu"], "navy": ["navy"],
    "purple": ["purple", "violet"], "pink": ["pink", "rose"],
}
_ALIAS2COLOR = {a: c for c, al in COLOR_ALIASES.items() for a in al}

# partial credit between neighbouring colors (light gray vs white, navy vs blue ...)
_RELATED = {
    "white": {"gray": .4, "beige": .4}, "gray": {"white": .4, "black": .4},
    "black": {"gray": .4, "navy": .3}, "brown": {"beige": .4, "orange": .4},
    "beige": {"white": .4, "brown": .4, "yellow": .2},
    "navy": {"blue": .7, "black": .3}, "blue": {"navy": .7},
    "red": {"pink": .2, "orange": .2}, "pink": {"red": .2, "purple": .2},
    "purple": {"pink": .3, "blue": .2},
    "orange": {"brown": .4, "yellow": .3, "red": .2},
    "yellow": {"orange": .3, "beige": .3}, "green": {},
}
COLOR_SIM = np.eye(len(COLOR_NAMES), dtype=np.float32)
for _c, _rel in _RELATED.items():
    for _o, _w in _rel.items():
        COLOR_SIM[_CIDX[_c], _CIDX[_o]] = _w

# ------------------------------------------------------------------ categories
CATEGORY_SYNONYMS = {
    "sofa": ["sofa", "couch", "loveseat", "settee", "sectional"],
    "chair": ["chair", "armchair", "stool", "recliner", "seat"],
    "table": ["table"],
    "desk": ["desk"],
    "bed": ["bed", "bunk"],
    "bookcase": ["bookcase", "bookshelf", "shelf", "shelves", "shelving"],
    "wardrobe": ["wardrobe", "closet", "cupboard", "armoire", "cabinet", "dresser"],
}
_SYN2CAT = {s: c for c, syns in CATEGORY_SYNONYMS.items() for s in syns}

# ------------------------------------------------------------------ text helpers
_VOCAB = sorted(set(_SYN2CAT) | set(_ALIAS2COLOR))
_NO_CORRECT = {"read", "reed", "wood", "would", "board", "tablet", "stable",
               "cable", "label", "bead", "teak", "oak", "wide", "wine", "while",
               "whole", "write", "wire", "brow", "rows", "tall", "tell"}
_TYPOS = {"sopha": "sofa", "soaf": "sofa", "coatch": "couch", "wardope": "wardrobe",
          "wardorbe": "wardrobe", "wardrop": "wardrobe", "wardobe": "wardrobe",
          "cahir": "chair", "chiar": "chair", "tabel": "table", "lmap": "lamp",
          "shef": "shelf", "cabnet": "cabinet", "bokcase": "bookcase"}


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z]+", (text or "").lower())


def _singular(w: str) -> str:
    if w.endswith("es") and w[:-2] in _SYN2CAT:
        return w[:-2]
    if w.endswith("s") and w[:-1] in _SYN2CAT:
        return w[:-1]
    return w


def correct_spelling(text: str) -> str:
    """Fuzzy-fix misspelled furniture/color words ('wardope'->'wardrobe')."""
    def repl(m):
        raw, w = m.group(), m.group().lower()
        if w in _TYPOS:
            return _TYPOS[w]
        if len(w) < 4 or w in _VOCAB or w in _NO_CORRECT or _singular(w) in _SYN2CAT:
            return raw
        cands = [v for v in _VOCAB if v[0] == w[0]]          # same first letter only
        hit = difflib.get_close_matches(w, cands, n=1, cutoff=0.75)
        return hit[0] if hit else raw
    return re.sub(r"[A-Za-z]+", repl, text or "")


def extract_color(text: str) -> Optional[str]:
    for w in _words(text):                      # whole words only
        if w in _ALIAS2COLOR:
            return _ALIAS2COLOR[w]
    return None


def extract_category(text: str) -> Optional[str]:
    for w in _words(text):
        w = _singular(w)
        if w in _SYN2CAT:
            return _SYN2CAT[w]
    return None


# ------------------------------------------------------------------ ranking
COLOR_W = 4.0        # z-units added for a perfect color match
CAT_BOOST = 2.0      # z-units added for the image-inferred category
MISC_PENALTY = 1.0   # Pix3D 'misc'/'tool' are noisy classes
# Typical RAW CLIP ViT-B/32 cosine ranges (heuristics - print text_cos/image_cos for a few
# good and bad queries on YOUR data and adjust). z-scores can't be used for confidence:
# the best of 10,000 random items always has z ~ 3.7, even for a nonsense query.
TEXT_COS_RANGE = (0.18, 0.32)    # text->image: unrelated ~0.15-0.20, good match ~0.28-0.34
IMAGE_COS_RANGE = (0.55, 0.92)   # image->image
LOW_CONF_REL = 0.35              # best relevance below this => low_confidence


def _rel(x: float, rng) -> float:
    lo, hi = rng
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def _z(s: np.ndarray) -> np.ndarray:
    return (s - s.mean()) / (s.std() + 1e-8)


class CatalogIndex:
    """Exact search over all catalog image embeddings (10k x 512 = ~20 MB)."""

    def __init__(self, matrix: np.ndarray, products_by_row: List[dict]):
        m = matrix.astype("float32")
        self.M = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)
        self.products = products_by_row
        self.cats = np.array([p.get("subCategory", "").lower() for p in products_by_row])
        H = np.zeros((len(products_by_row), len(COLOR_NAMES)), dtype=np.float32)
        for r, p in enumerate(products_by_row):
            hist = p.get("color_hist") or {p.get("baseColour", ""): 1.0}   # fallback to old label
            for name, frac in hist.items():
                if name in _CIDX:
                    H[r, _CIDX[name]] = frac
        self.H = H

    def color_scores(self, color: str) -> np.ndarray:
        return np.clip(self.H @ COLOR_SIM[_CIDX[color]], 0.0, 1.0)

    def search(self, text_vec=None, image_vec=None, color: Optional[str] = None,
               category: Optional[str] = None, alpha: float = 0.5, top_k: int = 10):
        if text_vec is None and image_vec is None:
            return [], {}
        s_t = self.M @ text_vec if text_vec is not None else None
        s_i = self.M @ image_vec if image_vec is not None else None

        if s_t is not None and s_i is not None:      # score-level fusion
            base = alpha * _z(s_t) + (1.0 - alpha) * _z(s_i)
        else:
            base = _z(s_t if s_t is not None else s_i)

        final = base.copy()
        meta = {"inferred_category": None}

        if category:                                  # hard filter (labels are reliable)
            final = np.where(self.cats == category, final, -np.inf)
        elif s_i is not None:                         # infer from nearest image neighbours
            top = np.argsort(-s_i)[:10]
            cat, n = Counter(self.cats[top]).most_common(1)[0]
            if n >= 6 and cat not in ("misc", "tool"):
                meta["inferred_category"] = cat
                final = final + CAT_BOOST * (self.cats == cat)
        if not category and not meta["inferred_category"]:
            final = final - MISC_PENALTY * np.isin(self.cats, ["misc", "tool"])

        cs = self.color_scores(color) if color else np.zeros(len(self.cats), np.float32)
        final = final + COLOR_W * cs

        order = np.argsort(-final)[:top_k]
        hits = []
        for r in order:
            if not np.isfinite(final[r]):
                continue
            rt = _rel(s_t[r], TEXT_COS_RANGE) if s_t is not None else None
            ri = _rel(s_i[r], IMAGE_COS_RANGE) if s_i is not None else None
            if rt is not None and ri is not None:
                rel = alpha * rt + (1.0 - alpha) * ri
            else:
                rel = rt if rt is not None else ri
            score = 0.85 * rel + 0.15 * float(cs[r]) if color else rel
            hits.append({
                "row": int(r),
                "score": float(score),
                "relevance": float(rel),
                "base_z": float(base[r]),
                "color_match": float(cs[r]),
                "text_cos": float(s_t[r]) if s_t is not None else None,
                "image_cos": float(s_i[r]) if s_i is not None else None,
            })
        meta["low_confidence"] = (not hits) or max(h["relevance"] for h in hits) < LOW_CONF_REL
        return hits, meta