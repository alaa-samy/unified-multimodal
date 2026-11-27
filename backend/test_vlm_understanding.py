"""Tests for the optional VLM understanding layer (no network, no torch).

    pytest backend/test_vlm_understanding.py
"""
import base64
import json

import vlm_understanding as vlm


# ------------------------------------------------------- disabled by default
def test_disabled_by_default(monkeypatch):
    monkeypatch.setattr(vlm, "VLM_PROVIDER", "")
    assert vlm.is_enabled() is False
    assert vlm.describe_image("whatever.jpg") is None  # no network call


# ------------------------------------------------------- parsing
def test_extract_json_block_from_chatty_answer():
    text = 'Sure! Here is the JSON: {"category": "sofa", "color": "black"} hope that helps'
    assert vlm._extract_json_block(text) == {"category": "sofa", "color": "black"}


def test_extract_json_block_rejects_garbage():
    assert vlm._extract_json_block("no json here at all") is None
    assert vlm._extract_json_block("{not valid json") is None
    assert vlm._extract_json_block("") is None


def test_normalize_attributes_drops_unknown_fillers():
    attrs = vlm._normalize_attributes({
        "category": "Sofa", "color": "BLACK", "material": "unknown",
        "pattern": "None", "style": "n/a", "caption": "A modern black leather sofa",
        "ignored": "field",
    })
    assert attrs == {"category": "sofa", "color": "black",
                     "caption": "a modern black leather sofa"}


def test_normalize_attributes_handles_non_strings():
    assert vlm._normalize_attributes({"color": 123, "category": None}) == {}


# ------------------------------------------------------- photo query build
def test_build_photo_query_orders_attributes():
    phrase = vlm.build_photo_query({"color": "black", "material": "leather",
                                    "pattern": "printed", "style": "modern",
                                    "category": "sofa", "caption": "x"})
    assert phrase == "black leather printed modern sofa"


def test_build_photo_query_category_can_be_excluded(monkeypatch):
    monkeypatch.setattr(vlm, "VLM_CATEGORY_FILTER", False)
    phrase = vlm.build_photo_query({"color": "black", "category": "sofa"})
    assert phrase == "black"


def test_build_photo_query_falls_back_to_caption():
    assert vlm.build_photo_query({"caption": "a round glass table"}) == "a round glass table"


# ------------------------------------------------------- ollama provider
def test_describe_ollama_parses_response(monkeypatch, tmp_path):
    img = tmp_path / "p.jpg"
    img.write_bytes(base64.b64decode(
        base64.b64encode(b"fake").decode()))  # tiny non-empty file

    class _R:
        def raise_for_status(self): pass
        def json(self):
            return {"response": '{"category": "chair", "color": "red", '
                               '"material": "wood", "pattern": "unknown", '
                               '"style": "vintage", "caption": "a red wooden chair"}'}

    captured = {}
    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["body"] = json
        return _R()

    monkeypatch.setattr(vlm.requests, "post", fake_post)
    monkeypatch.setattr(vlm, "VLM_PROVIDER", "ollama")
    monkeypatch.setattr(vlm, "OLLAMA_URL", "http://localhost:11434")

    out = vlm.describe_image(str(img))
    assert out == {"category": "chair", "color": "red", "material": "wood",
                   "style": "vintage", "caption": "a red wooden chair"}
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["body"]["model"] == vlm.VLM_OLLAMA_MODEL
    assert captured["body"]["images"][0] == base64.b64encode(b"fake").decode()


def test_describe_ollama_failure_returns_none(monkeypatch, tmp_path):
    img = tmp_path / "p.jpg"
    img.write_bytes(b"fake")

    def boom(*a, **k):
        raise ConnectionError("ollama down")

    monkeypatch.setattr(vlm.requests, "post", boom)
    monkeypatch.setattr(vlm, "VLM_PROVIDER", "ollama")
    assert vlm.describe_image(str(img)) is None


def test_describe_missing_file_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr(vlm, "VLM_PROVIDER", "ollama")
    assert vlm.describe_image(str(tmp_path / "nope.jpg")) is None
