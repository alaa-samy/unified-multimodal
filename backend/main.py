"""
Smart Furniture Search API
Run with: python main.py      (from the backend/ folder)

Setup (one time):
    pip install openai-whisper

Voice uses openai-whisper (free, local, no API key, works on Windows).
All models load lazily on first use. Server starts in under 2 seconds.

Requires search_core.py (same folder). Run `python recompute_metadata.py` once
to fix product colors (otherwise the old whole-image color labels are used).

=====================================================================
CHANGES IN THIS VERSION
=====================================================================
KEPT from your previous version
  - _is_color_modifier_only(): "similar but in black" + image => image-only
    embedding, color applied separately.
  - "fallback_tier" field in the response (see _do_search_with_fallback).
  - Whisper accuracy note (model is now configurable, default base.en).

NEW / CHANGED
  1. Text AND image queries are matched against the IMAGE embeddings. The
     TEXT index (built from names like "Brown Chair") is no longer searched.
  2. Text + image (+ voice = text) are fused at SCORE level with per-modality
     z-normalisation (alpha = text weight) instead of averaging two vectors.
  3. Color is a SOFT boost from a per-product color histogram, not a hard
     filter on one label. Whole-word matching ("standing" no longer = "tan").
  4. Category words (sofa/couch/closet...) are a hard filter; for image
     queries the category is inferred from nearest neighbours (soft boost).
  5. Fuzzy spell correction; Whisper initial prompt with furniture words.
  6. root() had lost its @app.get("/") decorator - restored.
  7. Response adds: corrected_query, category_filter, low_confidence.
=====================================================================
"""

import os
import re
import json
import shutil
import tempfile
import asyncio
import threading
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
import faiss
import torch
from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from search_core import (CatalogIndex, COLOR_ALIASES, extract_color,
                         extract_category, extract_material, extract_pattern,
                         correct_spelling)

import requests

# ============================================================
# Environment
# ============================================================

_base = Path(__file__).parent
for _f in [_base / ".env.local", _base / ".env"]:
    if _f.exists():
        load_dotenv(dotenv_path=str(_f), override=True)
        print(f"Loaded: {_f.name}")
        break

# ============================================================
# Config
# ============================================================

DATA_DIR         = _base / "data"
IMAGES_DIR       = DATA_DIR / "images"
MODELS_DIR       = DATA_DIR / "pix3d" / "glb"
PRODUCTS_PATH    = DATA_DIR / "clean_products.json"
IMAGE_INDEX_PATH = DATA_DIR / "image_index.faiss"
ID_MAP_PATH      = DATA_DIR / "id_map.npy"
BASE_URL         = os.getenv("BASE_URL", "http://localhost:8000")
WHISPER_MODEL    = os.getenv("WHISPER_MODEL", "base.en")   # "tiny" mis-hears short words
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NEW: LLM query understanding (the proposal's "Unified Understanding Engine").
# A local Ollama LLM rewrites conversational queries into a compact,
# attribute-rich search phrase so CLIP embeds the RIGHT thing.
# Option A (cloud, nothing to download - RECOMMENDED): free Groq API
#   -> set LLM_API_BASE + LLM_API_KEY + LLM_QUERY_MODEL (OpenAI-compatible)
# Option B (local): Ollama -> set OLLAMA_URL (default http://localhost:11434)
OLLAMA_URL       = os.getenv("OLLAMA_URL", "http://localhost:11434")
LLM_QUERY_MODEL  = os.getenv("LLM_QUERY_MODEL", "llama3")
LLM_API_BASE     = os.getenv("LLM_API_BASE", "")   # e.g. https://api.groq.com/openai/v1
LLM_API_KEY      = os.getenv("LLM_API_KEY", "")

# CLIP does better with a prompt than a bare word; average a few templates
TEXT_TEMPLATES = ["a photo of {}", "a photo of a {}, furniture", "{}"]
WHISPER_HINT   = ("Furniture search: sofa, couch, chair, table, desk, bed, bookcase, "
                  "bookshelf, wardrobe, closet, black, white, gray, brown, beige, navy, blue, red.")

# ============================================================
# State
# ============================================================

_products      = None
_catalog: Optional[CatalogIndex] = None
_clip_model    = None
_clip_proc     = None
_whisper_model = None

_data_ready    = False
_clip_ready    = False
_whisper_ready = False
_data_error    = None
_clip_error    = None
_whisper_error = None

_data_lock    = threading.Lock()
_clip_lock    = threading.Lock()
_whisper_lock = threading.Lock()

# ============================================================
# App
# ============================================================

app = FastAPI(title="Multimodal Furniture Search", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


class _CORSStatic(StaticFiles):
    async def get_response(self, path, scope):
        r = await super().get_response(path, scope)
        r.headers["Access-Control-Allow-Origin"] = "*"
        r.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        return r


if IMAGES_DIR.exists():
    app.mount("/static/images", _CORSStatic(directory=str(IMAGES_DIR)), name="images")
if MODELS_DIR.exists():
    app.mount("/static/models", _CORSStatic(directory=str(MODELS_DIR)), name="models")


@app.on_event("startup")
async def startup():
    print("\n=== Server ready: http://localhost:8000 ===\n")

# ============================================================
# Lazy loaders
# ============================================================

def _ensure_data():
    global _products, _catalog, _data_ready, _data_error
    with _data_lock:
        if _data_ready:
            return True
        try:
            print("[DATA] Loading...")
            with open(PRODUCTS_PATH, encoding="utf-8") as f:
                _products = json.load(f)
            index  = faiss.read_index(str(IMAGE_INDEX_PATH))
            id_map = np.load(str(ID_MAP_PATH))
            matrix = index.reconstruct_n(0, index.ntotal)       # exact search in numpy
            rows   = [_products[int(pid)] for pid in id_map]    # row i <-> product
            _catalog = CatalogIndex(matrix, rows)
            _data_ready = True
            n_hist = sum(1 for p in _products if p.get("color_hist"))
            print(f"[DATA] Ready - {len(_products)} products, {n_hist} with color histograms")
            if n_hist == 0:
                print("[DATA] WARNING: run `python recompute_metadata.py` to fix colors")
            return True
        except Exception as e:
            _data_error = str(e)
            print(f"[DATA] ERROR: {e}")
            traceback.print_exc()
            return False


def _ensure_clip():
    global _clip_model, _clip_proc, _clip_ready, _clip_error
    with _clip_lock:
        if _clip_ready:
            return True
        try:
            print("[CLIP] Loading... (~30s first time)")
            from transformers import CLIPModel, CLIPProcessor
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            _clip_model.to(DEVICE).eval()
            _clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            _clip_ready = True
            print("[CLIP] Ready")
            return True
        except Exception as e:
            _clip_error = str(e)
            print(f"[CLIP] ERROR: {e}")
            traceback.print_exc()
            return False


def _ensure_whisper():
    """
    Uses openai-whisper (original package, NOT faster-whisper).
    Install: pip install openai-whisper
    "tiny" is fast but unreliable on short utterances (e.g. "black" -> "a button
    back"), so the default is now "base.en". Override with env WHISPER_MODEL.
    """
    global _whisper_model, _whisper_ready, _whisper_error
    with _whisper_lock:
        if _whisper_ready:
            return True
        try:
            print(f"[WHISPER] Loading openai-whisper {WHISPER_MODEL}...")
            import whisper
            _whisper_model = whisper.load_model(WHISPER_MODEL)
            _whisper_ready = True
            print("[WHISPER] Ready")
            return True
        except ImportError:
            _whisper_error = "openai-whisper not installed. Run: pip install openai-whisper"
            print(f"[WHISPER] {_whisper_error}")
            return False
        except Exception as e:
            _whisper_error = str(e)
            print(f"[WHISPER] ERROR: {e}")
            traceback.print_exc()
            return False

# ============================================================
# Color-modifier detection (kept from previous version)
# ============================================================

_MODIFIER_STOPWORDS = {
    "i", "want", "to", "search", "for", "similar", "this", "image", "but", "in",
    "give", "me", "a", "the", "find", "show", "looking", "that", "is", "with",
    "product", "item", "style", "one", "please", "can", "you", "maybe",
    "different", "color", "colour", "also", "instead", "or", "same", "just", "of",
    "it", "make", "another", "other", "version", "like", "its", "look", "looks",
}


def _is_color_modifier_only(text: str, color: Optional[str]) -> bool:
    """
    True if, after stripping generic instruction phrasing and the detected
    color word, nothing substantive is left ("...but in black"). In that case
    the text is NOT embedded with the image (instruction words like
    "search"/"similar" would dilute the visual signal); the color is applied
    as a boost instead.
    """
    if not text or not color:
        return False
    color_variants = set(COLOR_ALIASES.get(color, [color]))
    leftover = []
    for w in text.split():
        w = w.strip(".,!?\"'").lower()
        if w and w not in _MODIFIER_STOPWORDS and w not in color_variants:
            leftover.append(w)
    return len(leftover) == 0

# ============================================================
# Encoding + search (blocking, called via run_in_executor)
# ============================================================

def _normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def _encode_text(q: str) -> np.ndarray:
    prompts = [t.format(q) for t in TEXT_TEMPLATES]
    inp = _clip_proc(text=prompts, return_tensors="pt", padding=True,
                     truncation=True, max_length=77).to(DEVICE)
    with torch.no_grad():
        f = _clip_model.get_text_features(**inp).cpu().numpy().astype("float32")
    return _normalize(_normalize(f).mean(axis=0))


def _encode_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    inp = _clip_proc(images=img, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        f = _clip_model.get_image_features(**inp).cpu().numpy().astype("float32")
    return _normalize(f[0])


def _encode_images(paths: Optional[List[str]]) -> Optional[np.ndarray]:
    """
    NEW: multi-image support. Encode each uploaded image, then fuse them into
    a single unit vector by averaging the normalized embeddings (and
    re-normalizing). A single image is returned as-is; failed images are
    skipped so one bad upload doesn't kill the whole search.
    """
    if not paths:
        return None
    vecs = []
    for p in paths:
        try:
            vecs.append(_encode_image(p))
        except Exception as e:
            print(f"[CLIP] Failed to encode image {p}: {e}")
    if not vecs:
        return None
    if len(vecs) == 1:
        return vecs[0]
    fused = np.mean(np.stack(vecs, axis=0), axis=0)
    return _normalize(fused)


def _make_result(product: dict, hit: dict) -> dict:
    img  = product.get("image_path", "")
    glb  = product.get("glb_path")
    has3 = product.get("has_3d_model", False)
    return {
        "id":           product["id"],
        "name":         product["productDisplayName"],
        "category":     product.get("masterCategory", "Furniture"),
        "subCategory":  product.get("subCategory", ""),
        "image":        img,
        "color":        product.get("baseColour", ""),
        "score":        round(hit["score"], 4),
        "color_match":  round(hit["color_match"], 3),
        "text_cos":     hit["text_cos"],
        "image_cos":    hit["image_cos"],
        "has_3d_model": has3,
        "image_url":    f"{BASE_URL}/static/images/{img}" if img else "",
        "model_url":    f"{BASE_URL}/static/models/{glb}" if (has3 and glb) else None,
    }


def _do_search(query: str, img_paths: Optional[List[str]], color: Optional[str],
               category: Optional[str], alpha: float, top_k: int):
    """Encode text/images, fuse scores, rank. Returns (results, meta)."""
    t = _encode_text(query) if query.strip() else None
    i = _encode_images(img_paths)
    hits, meta = _catalog.search(t, i, color=color, category=category,
                                 alpha=alpha, top_k=top_k)
    return [_make_result(_catalog.products[h["row"]], h) for h in hits], meta


_MIN_COLOR_MATCHES = 3  # fewer color-matching results than this => tier "color_relaxed"


def _do_search_with_fallback(query: str, img_paths: Optional[List[str]],
                             color: Optional[str], category: Optional[str],
                             alpha: float, top_k: int):
    """
    Returns (results, tier, meta). Color is now a soft boost, so a single search
    already ranks color matches first and still returns the best of the rest
    when the catalog has too few (e.g. "red sofa" with no red sofas). The tier
    just tells the UI which case happened:
      - "exact"          : no color requested, or >= 3 results match it
      - "color_relaxed"  : right category/visuals, but the catalog has few/no
                           items in the requested color
      - "none"           : nothing found
    (The old "category_only" tier is gone: the category is a hard filter now,
    so there is nothing left to broaden.)
    """
    results, meta = _do_search(query, img_paths, color, category, alpha, top_k)
    if not results:
        return [], "none", meta
    if not color:
        return results, "exact", meta
    matched = sum(1 for r in results if r["color_match"] >= 0.5)
    tier = "exact" if matched >= min(_MIN_COLOR_MATCHES, len(results)) else "color_relaxed"
    return results, tier, meta


_HALLUCINATIONS = {"thank you", "thanks for watching", "thank you for watching", "you", "bye", "the end"}
_STYLE_WORDS = {"furniture", "wooden", "wood", "leather", "fabric", "modern", "vintage", "small",
                "large", "big", "dark", "cheap", "metal", "glass", "comfortable",
                # NEW: material / pattern words count as a valid search signal
                "printed", "printted", "pattern", "patterned", "striped", "floral",
                "velvet", "linen", "rattan", "wicker", "plaid", "chrome", "oak"}


def _rewrite_query_with_llm(query: str) -> str:
    """
    NEW: LLM-based query understanding (proposal section 7.1.2, "Unified
    Understanding Engine"). Rewrites a conversational request into a short,
    attribute-rich search phrase that keeps EVERY visual attribute
    (color, material, pattern, style, category):

        "i want black leather sofa but with printted pattern"
            -> "black leather sofa with printed pattern"

    CLIP then embeds a clean phrase instead of a chatty sentence, so
    material/pattern actually influence the ranking. Falls back silently to
    the raw query when Ollama is not running.
    """
    if not query.strip():
        return query
    prompt = (
        "You rewrite furniture shopping requests into product search phrases. "
        "Rules:\n"
        "1. Keep ALL visual attributes: color, material (leather, velvet, wood, "
        "metal...), pattern (printed, floral, striped, plaid...), style and the "
        "furniture category.\n"
        "2. Fix spelling (e.g. 'printted' -> 'printed').\n"
        "3. Remove filler words like 'i want', 'but', 'please', 'find me'.\n"
        "4. Maximum 12 words, lowercase, no punctuation.\n"
        "5. Output ONLY the search phrase, nothing else.\n\n"
        f"Request: {query}\nSearch phrase:"
    )
    try:
        if LLM_API_BASE and LLM_API_KEY:
            # Cloud option (Groq / any OpenAI-compatible API) - nothing to download
            r = requests.post(
                f"{LLM_API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                json={"model": LLM_QUERY_MODEL,
                      "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0},
                timeout=8,
            )
            out = (r.json()["choices"][0]["message"]["content"] or "").strip()
        else:
            # Local option (Ollama)
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": LLM_QUERY_MODEL, "prompt": prompt,
                      "stream": False, "options": {"temperature": 0}},
                timeout=8,
            )
            out = (r.json().get("response") or "").strip()
        out = out.strip('"').strip(".").lower()
        words = out.split()
        if words and len(words) <= 15:
            if out != query.strip().lower():
                print(f"[LLM] Query rewrite: {query!r} -> {out!r}")
            return out
        print(f"[LLM] Rewrite ignored (bad output: {out!r})")
    except Exception as e:
        print(f"[LLM] Rewrite unavailable ({e}) - using raw query. "
              f"Set LLM_API_BASE/LLM_API_KEY (cloud) or start Ollama for best results.")
    return query


def _looks_like_search(text: str) -> bool:
    """Whole-word check (the old substring check matched 'show' inside 'showed')."""
    toks = re.findall(r"[a-z]+", (text or "").lower())
    return bool(extract_category(text) or extract_color(text) or any(t in _STYLE_WORDS for t in toks))


def _transcribe_audio(path: str) -> dict:
    """
    Read the browser-produced 16 kHz mono WAV with the stdlib and run Whisper.
    Returns text + diagnostics so the UI/you can tell a bad recording from a bad model.
    """
    out = {"text": "", "error": None, "warning": None, "duration": 0.0, "rms": 0.0,
           "peak": 0.0, "no_speech_prob": None, "avg_logprob": None}
    if not _whisper_model:
        out["error"] = "Whisper unavailable"
        return out
    try:
        import wave
        with wave.open(path, "rb") as wf:
            n_channels, sampwidth = wf.getnchannels(), wf.getsampwidth()
            framerate, raw = wf.getframerate(), wf.readframes(wf.getnframes())
        if sampwidth == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)
        if framerate != 16000 and len(samples) > 1:
            new_len = int(len(samples) * 16000 / framerate)
            samples = np.interp(np.linspace(0, len(samples) - 1, new_len),
                                np.arange(len(samples)), samples).astype(np.float32)

        n = len(samples)
        out["duration"] = round(n / 16000.0, 2)
        out["peak"] = round(float(np.abs(samples).max()), 4) if n else 0.0
        out["rms"] = round(float(np.sqrt(np.mean(samples ** 2))), 5) if n else 0.0

        if os.getenv("DEBUG_SAVE_AUDIO") == "1":          # play it back to check the mic path
            shutil.copy(path, _base / "debug_last_audio.wav")
        if out["duration"] < 0.4:
            out["error"] = "Recording too short - speak for at least a second."
            return out
        if out["peak"] < 0.01:
            out["error"] = "Recording is silent - check the selected microphone and browser permission."
            return out
        if out["peak"] < 0.5:                             # boost quiet mics (max 20x)
            samples = (samples * min(0.9 / out["peak"], 20.0)).astype(np.float32)

        res = _whisper_model.transcribe(
            samples, language="en", fp16=False, temperature=0.0,
            condition_on_previous_text=False, no_speech_threshold=0.6,
            initial_prompt=WHISPER_HINT)
        text = res["text"].strip()
        segs = res.get("segments") or []
        if segs:
            out["no_speech_prob"] = round(float(np.mean([s["no_speech_prob"] for s in segs])), 3)
            out["avg_logprob"] = round(float(np.mean([s["avg_logprob"] for s in segs])), 3)

        norm = re.sub(r"[^a-z ]", "", text.lower()).strip()
        if norm.startswith("furniture search") or norm in _HALLUCINATIONS:
            text = ""                                      # Whisper echoed the prompt / filler
        if not text:
            out["error"] = "Could not understand audio - please try again."
        elif ((out["no_speech_prob"] or 0) > 0.6) or ((out["avg_logprob"] or 0) < -1.0):
            out["warning"] = "Low confidence - please check the text."
        out["text"] = text
        print(f"[WHISPER] Heard: {text!r} | {out}")
        return out
    except Exception as e:
        print(f"[WHISPER] error: {e}")
        traceback.print_exc()
        out["error"] = f"Transcription failed: {e}"
        return out


# ============================================================
# Routes
# ============================================================



@app.get("/")
def root():
    return {"status": "online", "data_ready": _data_ready,
            "clip_ready": _clip_ready, "whisper_ready": _whisper_ready,
            "data_error": _data_error, "clip_error": _clip_error,
            "whisper_error": _whisper_error,
            "products": len(_products) if _products else 0}


@app.get("/health")
def health():
    return {"status": "healthy" if (_data_ready and _clip_ready) else "loading",
            "data_ready": _data_ready, "clip_ready": _clip_ready,
            "whisper_ready": _whisper_ready,
            "data_error": _data_error, "clip_error": _clip_error,
            "whisper_error": _whisper_error,
            "products": len(_products) if _products else 0}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Voice -> text only, so the UI can show/edit what was heard BEFORE searching."""
    loop = asyncio.get_event_loop()
    data = await audio.read()
    if len(data) < 200:
        return {"ok": False, "text": "", "error": "Recording too short - speak for at least a second."}
    if not _whisper_ready:
        await loop.run_in_executor(None, _ensure_whisper)
    if not _whisper_ready:
        return {"ok": False, "text": "", "error": _whisper_error or "Whisper unavailable"}
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(data)
            tmp = f.name
        r = await loop.run_in_executor(None, _transcribe_audio, tmp)
    finally:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
    text = correct_spelling(r["text"]) if r["text"] else ""
    looks = _looks_like_search(text) if text else False
    warning = r["warning"]
    if text and not looks:
        warning = "This doesn't sound like a furniture search - edit the text or record again."
    return {"ok": bool(text) and not r["error"], "text": text, "raw_text": r["text"],
            "error": r["error"], "warning": warning, "looks_like_search": looks,
            "diagnostics": {k: r[k] for k in ("duration", "rms", "peak", "no_speech_prob", "avg_logprob")}}


@app.get("/products/{product_id}")
async def get_product(product_id: int):
    loop = asyncio.get_event_loop()
    if not _data_ready:
        await loop.run_in_executor(None, _ensure_data)
    if not _data_ready:
        raise HTTPException(503, detail=f"Data error: {_data_error}")
    if product_id < 0 or product_id >= len(_products):
        raise HTTPException(404, detail="Product not found")
    p    = _products[product_id]
    glb  = p.get("glb_path")
    has3 = p.get("has_3d_model", False)
    img  = p.get("image_path", "")
    murl = f"{BASE_URL}/static/models/{glb}" if (has3 and glb) else None
    return {
        "id": p["id"], "name": p["productDisplayName"],
        "category": p.get("masterCategory", "Furniture"),
        "subCategory": p.get("subCategory", ""),
        "image": img, "color": p.get("baseColour", ""),
        "has_3d_model": has3, "glb_path": glb or "",
        "image_url": f"{BASE_URL}/static/images/{img}" if img else "",
        "model_url": murl,
        "model_exists": (MODELS_DIR / glb).exists() if murl else False,
    }


@app.post("/search/intelligent")
async def search(
    query: Optional[str] = Form(""),
    file:  Optional[UploadFile] = File(None),        # legacy single-image field
    files: Optional[List[UploadFile]] = File(None), # NEW: multiple images
    image: Optional[UploadFile] = File(None),       # legacy alias
    audio: Optional[UploadFile] = File(None),
    top_k: int = Form(10),
    alpha: float = Form(0.5),     # text weight when text AND image are both used
    voice_text: Optional[str] = Form(""),   # transcript already shown/edited in the UI
):
    loop = asyncio.get_event_loop()

    if not _data_ready:
        await loop.run_in_executor(None, _ensure_data)
    if not _data_ready:
        raise HTTPException(503, detail=f"Data error: {_data_error}")

    if not _clip_ready:
        print("[CLIP] First request - loading now (~30s)...")
        ok = await loop.run_in_executor(None, _ensure_clip)
        if not ok:
            raise HTTPException(500, detail=f"CLIP failed: {_clip_error}")

    typed_query = (query or "").strip()
    voice_query = ""
    voice_error = None
    tmp_audio   = None
    tmp_imgs    = []          # NEW: one temp path per uploaded image

    # ---------------- voice -> text ----------------
    voice_rejected = None
    voice_text = (voice_text or "").strip()
    if voice_text:
        voice_query = voice_text            # user already saw / edited this in the UI
        print(f"[VOICE] Using confirmed transcript: {voice_query!r}")
    elif audio:
        try:
            data = await audio.read()
            print(f"[VOICE] {len(data)} bytes")
            if not _whisper_ready:
                await loop.run_in_executor(None, _ensure_whisper)
            if not _whisper_ready:
                voice_error = _whisper_error or "Whisper unavailable"
            elif len(data) < 200:
                voice_error = "Audio too short - hold mic button longer"
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    f.write(data)
                    tmp_audio = f.name
                r = await loop.run_in_executor(None, _transcribe_audio, tmp_audio)
                if r["error"]:
                    voice_error = r["error"]
                elif not _looks_like_search(r["text"]):
                    voice_rejected = r["text"]
                    voice_error = (f"I heard \"{r['text']}\", which doesn't sound like a "
                                   f"furniture search. Please record again.")
                    print(f"[VOICE] Rejected (no furniture/color words): {r['text']!r}")
                else:
                    voice_query = r["text"]
        except Exception as e:
            voice_error = str(e)
            print(f"[VOICE] Error: {e}")
        finally:
            if tmp_audio and os.path.exists(tmp_audio):
                os.remove(tmp_audio)

    # ---------------- merge text sources + spell correction ----------------
    final_query = " ".join(x for x in (voice_query, typed_query) if x).strip()
    corrected_query = None
    if final_query:
        fixed = correct_spelling(final_query)
        if fixed.lower() != final_query.lower():
            corrected_query = fixed
            print(f"[SEARCH] Spell corrected: {final_query!r} -> {fixed!r}")
            final_query = fixed

    # ---------------- LLM query understanding (NEW) ----------------
    # "i want black leather sofa but with printted pattern"
    #   -> "black leather sofa with printed pattern"
    # Falls back to the raw query if Ollama is not running.
    interpreted_query = final_query
    if final_query:
        rewritten = _rewrite_query_with_llm(final_query)
        if rewritten and rewritten.lower() != final_query.lower():
            interpreted_query = rewritten
            final_query = rewritten

    color_filter     = extract_color(final_query)
    category_filter  = extract_category(final_query)
    material_filter  = extract_material(final_query)   # NEW: leather / velvet / wood ...
    pattern_filter   = extract_pattern(final_query)   # NEW: printed / striped / floral ...

    print(f"[SEARCH] typed={typed_query!r} voice={voice_query!r} final={final_query!r} "
          f"color={color_filter} category={category_filter} "
          f"material={material_filter} pattern={pattern_filter}")

    # ---------------- image(s) ----------------
    # Collect uploads from every supported field name (legacy single "file"/"image"
    # plus the new repeated "files" field). Order is preserved.
    img_uploads = [f for f in ([file] + list(files or []) + [image]) if f is not None]
    for img_upload in img_uploads:
        try:
            data = await img_upload.read()
            if not data:
                continue
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
                f.write(data)
                tmp_imgs.append(f.name)
        except Exception as e:
            print(f"[IMAGE] Error saving upload: {e}")
    img_count = len(tmp_imgs)
    if img_count > 1:
        print(f"[SEARCH] {img_count} images uploaded - fusing embeddings")

    if not final_query and not tmp_imgs:
        return _empty(typed_query, voice_query, voice_error,
                      "Please provide text, image, or voice")

    try:
        # Image + pure color/style instruction ("similar but in black"): embed the
        # image only; the color is applied as a boost by the ranker.
        search_query = final_query
        if tmp_imgs and color_filter and _is_color_modifier_only(final_query, color_filter):
            print(f"[SEARCH] {final_query!r} is color-modifier-only - "
                  f"image-only embedding + {color_filter!r} boost")
            search_query = ""

        # NEW: when the query names a material or pattern, the TEXT signal is
        # the only place that information lives (the catalog metadata has no
        # material/pattern labels), so give it more weight in the fusion.
        if (material_filter or pattern_filter) and search_query:
            alpha = min(0.75, alpha + 0.25)
            print(f"[SEARCH] Specific attributes "
                  f"(material={material_filter}, pattern={pattern_filter}) - "
                  f"raising text weight to alpha={alpha:.2f}")

        results, fallback_tier, meta = await loop.run_in_executor(
            None, _do_search_with_fallback, search_query, tmp_imgs, color_filter,
            category_filter, max(0.0, min(1.0, alpha)), top_k)
    except Exception as e:
        print(f"[SEARCH] ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(500, detail=f"Search failed: {e}")
    finally:
        for tmp in tmp_imgs:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)

    parts = (["text"] if typed_query else []) + \
            ((["image"] if img_count == 1 else ["images"]) if img_count else []) + \
            (["voice"] if voice_query else [])
    mode      = " + ".join(parts) if parts else "unknown"
    color_str = f" {color_filter}" if color_filter else ""

    return {
        "query": final_query, "interpreted_query": interpreted_query,
        "corrected_query": corrected_query,
        "typed_query": typed_query, "voice_query": voice_query,
        "voice_error": voice_error, "voice_rejected_text": voice_rejected,
        "accurate_results": results, "related_results": [],
        "total_results": len(results), "color_filter": color_filter,
        "category_filter": category_filter or meta.get("inferred_category"),
        "material_filter": material_filter, "pattern_filter": pattern_filter,
        "low_confidence": meta.get("low_confidence", False),
        "search_mode": mode,
        "fallback_tier": fallback_tier,
        "ai_summary": f"Found {len(results)}{color_str} items matching "
                      f"'{final_query}' [{mode}]",
    }


def _empty(typed, voice, verr, msg):
    return {"query": "", "interpreted_query": "", "corrected_query": None,
            "typed_query": typed, "voice_query": voice, "voice_error": verr, "voice_rejected_text": None,
            "accurate_results": [], "related_results": [],
            "total_results": 0, "color_filter": None, "category_filter": None,
            "material_filter": None, "pattern_filter": None,
            "low_confidence": True, "search_mode": "none",
            "fallback_tier": "none", "ai_summary": msg}


if __name__ == "__main__":
    import uvicorn
    print("\nStarting - http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False, workers=1)