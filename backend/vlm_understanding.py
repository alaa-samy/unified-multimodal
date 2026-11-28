"""
vlm_understanding.py  -  optional Vision-Language understanding layer
======================================================================

Implements the proposal's multimodal-model commitment (proposal §7.2.1):

    "Select and implement base multimodal models like: CLIP for
    visual-semantic embeddings, LLaVA for detailed visual understanding
    and CogVLM for complex visual reasoning."

Role in the pipeline
--------------------
CLIP (always on)            -> embeddings + retrieval (search_core.py)
LLM query rewrite (optional) -> text-only conversational cleanup (main.py)
VLM understanding (THIS)     -> looks INSIDE an uploaded product photo and
                                turns it into searchable attributes:
                                {"category", "color", "material", "pattern",
                                 "style", "caption"}

When is it used?
----------------
Only for PHOTO-ONLY searches (the user uploaded an image and typed/said
nothing). Today such a search relies purely on the CLIP image embedding;
with the VLM enabled, the photo is also described in words and the words are
encoded as a text query, fused with the image embedding by the existing
z-normalised score fusion. Image+text and voice searches are never modified,
so the deterministic pipeline stays byte-identical when the VLM is off.

Providers (all free)
--------------------
VLM_PROVIDER = "ollama"  -> local Ollama running a vision model
                            (`ollama pull llava:7b`), default
                            LLaVA - "detailed visual understanding".
                            No GPU service, no cost, nothing leaves the machine.
VLM_PROVIDER = "cogvlm"  -> local HuggingFace transformers CogVLM adapter
                            ("complex visual reasoning"). Heavy (needs a GPU
                            and trust_remote_code); loaded lazily only when
                            selected, so it never touches the default path.
VLM_PROVIDER = ""        -> disabled (default). describe_image() returns None
                            without any network call.

Failure policy
--------------
Any error (model missing, timeout, malformed JSON, out-of-vocabulary values)
returns None / drops the field - the search then behaves exactly as it would
without the VLM. The raw description is attached to the API response as
"vlm_understanding" so the evaluation dashboard can audit every VLM-assisted
search.
"""
import base64
import json
import os
import re
from typing import Any, Dict, List, Optional

import requests

VLM_PROVIDER = os.getenv("VLM_PROVIDER", "").strip().lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
VLM_OLLAMA_MODEL = os.getenv("VLM_OLLAMA_MODEL", "llava:7b")
COGVLM_MODEL_ID = os.getenv("COGVLM_MODEL_ID", "THUDM/cogvlm-base-224")
# When "0", the VLM-derived category is NOT applied as the hard category
# filter; the category stays inferred from the CLIP image neighbours instead.
VLM_CATEGORY_FILTER = os.getenv("VLM_CATEGORY_FILTER", "1") == "1"
VLM_TIMEOUT = int(os.getenv("VLM_TIMEOUT", "30"))

_PROMPT = (
    "Look at this furniture product photo. Answer with ONLY a JSON object, "
    "no other text, using these exact keys:\n"
    '{"category": "<one word: sofa, chair, table, desk, bed, bookcase, '
    'wardrobe, or other>", '
    '"color": "<dominant color, one word>", '
    '"material": "<wood, metal, leather, fabric, glass, or unknown>", '
    '"pattern": "<printed, striped, plaid, solid, or unknown>", '
    '"style": "<modern, vintage, rustic, minimalist, classic, or unknown>", '
    '"caption": "<one short sentence describing the item>"}'
)


def is_enabled() -> bool:
    return VLM_PROVIDER in ("ollama", "cogvlm")


# ------------------------------------------------------------------ parsing
def _extract_json_block(text: str) -> Optional[dict]:
    """Pull the first JSON object out of a chatty VLM answer."""
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


_UNKNOWN = {"", "unknown", "none", "null", "n/a", "na", "other", "unspecified"}


def _clean(value: Any, max_words: int = 3) -> Optional[str]:
    """Normalise one attribute string; drop the VLM's 'unknown' filler."""
    if not isinstance(value, str):
        return None
    words = re.findall(r"[a-zA-Z]+", value.lower())
    if not words:
        return None
    joined = " ".join(words[:max_words])
    # "n/a" splits into "n a" - compare with spaces stripped as well
    if joined in _UNKNOWN or joined.replace(" ", "") in _UNKNOWN:
        return None
    return joined


def _normalize_attributes(raw: dict) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for key in ("category", "color", "material", "pattern", "style"):
        cleaned = _clean(raw.get(key))
        if cleaned:
            attrs[key] = cleaned
    caption = _clean(raw.get("caption"), max_words=15)
    if caption:
        attrs["caption"] = caption
    return attrs


def build_photo_query(description: Dict[str, str]) -> str:
    """
    Turn a normalised VLM description into a short search phrase, e.g.
    {"color": "black", "material": "leather", "category": "sofa"}
        -> "black leather sofa"
    The caption is the last-resort phrase when no attributes survived.
    """
    parts = [description.get(k) for k in
             ("color", "material", "pattern", "style") if description.get(k)]
    if VLM_CATEGORY_FILTER and description.get("category"):
        parts.append(description["category"])
    phrase = " ".join(parts)
    if not phrase:
        phrase = description.get("caption", "")
    return " ".join(phrase.split()[:15])


# ------------------------------------------------------------------ providers
def _describe_ollama(image_path: str) -> Optional[Dict[str, str]]:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    r = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": VLM_OLLAMA_MODEL, "prompt": _PROMPT,
              "images": [b64], "stream": False,
              "options": {"temperature": 0}},
        timeout=VLM_TIMEOUT,
    )
    r.raise_for_status()
    raw = _extract_json_block(r.json().get("response") or "")
    return _normalize_attributes(raw) if raw else None


_cogvlm = None  # lazy singleton


def _describe_cogvlm(image_path: str) -> Optional[Dict[str, str]]:
    """CogVLM adapter ('complex visual reasoning'). Needs torch + transformers
    with trust_remote_code and a GPU; loaded lazily and only when selected."""
    global _cogvlm
    try:
        from PIL import Image as PILImage

        if _cogvlm is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(COGVLM_MODEL_ID,
                                                trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                COGVLM_MODEL_ID, trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            _cogvlm = (tok, model)

        tok, model = _cogvlm
        image = PILImage.open(image_path).convert("RGB")
        input_by_model = model.build_conversation_input_ids(
            tok, query=_PROMPT, history=[], images=[image])
        import torch

        inputs = {
            "input_ids": input_by_model["input_ids"].unsqueeze(0).to(model.device),
            "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to(model.device),
            "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to(model.device),
            "images": [[input_by_model["images"][0].to(model.device).to(torch.float16
                        if torch.cuda.is_available() else torch.float32)]],
        }
        gen_kwargs = {"max_new_tokens": 128, "do_sample": False}
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        answer = tok.decode(outputs[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)
        raw = _extract_json_block(answer)
        return _normalize_attributes(raw) if raw else None
    except Exception as exc:  # model missing, no GPU, bad output...
        print(f"[VLM] CogVLM adapter unavailable ({exc}) - continuing without it")
        return None


# ------------------------------------------------------------------ public API
def describe_image(image_path: str) -> Optional[Dict[str, str]]:
    """Describe one product photo. Returns {"category":..., "color":...,
    "material":..., "pattern":..., "style":..., "caption":...} or None when
    the VLM is disabled or anything fails (the pipeline then runs unchanged)."""
    if not is_enabled() or not image_path or not os.path.exists(image_path):
        return None
    try:
        if VLM_PROVIDER == "ollama":
            return _describe_ollama(image_path)
        if VLM_PROVIDER == "cogvlm":
            return _describe_cogvlm(image_path)
    except Exception as exc:
        print(f"[VLM] Description failed ({exc}) - continuing without it")
    return None
