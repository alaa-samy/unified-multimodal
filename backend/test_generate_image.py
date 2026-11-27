"""Tests for POST /generate_image. The external image service is replaced by a stub."""
import io

import pytest
import requests
from fastapi.testclient import TestClient
from PIL import Image

import main


def _png(size=(16, 16)):
    buf = io.BytesIO()
    Image.new("RGB", size, (150, 30, 30)).save(buf, "PNG")
    return buf.getvalue()


class _Resp:
    def __init__(self, content=b"", status=200, ctype="image/png"):
        self.content, self.status_code, self.headers = content, status, {"content-type": ctype}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "GENERATED_DIR", tmp_path)
    return TestClient(main.app), tmp_path


def test_success_stores_image_and_returns_url(client, monkeypatch):
    c, out = client
    monkeypatch.setattr(main.requests, "get", lambda *a, **k: _Resp(_png()))
    r = c.post("/generate_image", data={"prompt": "a red velvet sofa"})
    assert r.status_code == 200
    body = r.json()
    assert body["prompt"] == "a red velvet sofa"
    assert "/static/generated/" in body["image_url"]
    assert len(list(out.glob("*.jpg"))) == 1


def test_prompt_whitespace_is_normalised(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr(main.requests, "get", lambda *a, **k: _Resp(_png()))
    r = c.post("/generate_image", data={"prompt": "  a   red\n sofa  "})
    assert r.json()["prompt"] == "a red sofa"


@pytest.mark.parametrize("resp", [_Resp(b"<html>", 200, "text/html"), _Resp(b"x", 500),
                                  _Resp(b"not an image", 200, "image/png")])
def test_bad_upstream_responses_are_502(client, monkeypatch, resp):
    c, out = client
    monkeypatch.setattr(main.requests, "get", lambda *a, **k: resp)
    r = c.post("/generate_image", data={"prompt": "a red velvet sofa"})
    assert r.status_code == 502
    assert list(out.glob("*.jpg")) == []


def test_unreachable_service_is_502(client, monkeypatch):
    c, _ = client
    def boom(*a, **k):
        raise requests.ConnectionError("down")
    monkeypatch.setattr(main.requests, "get", boom)
    assert c.post("/generate_image", data={"prompt": "a red velvet sofa"}).status_code == 502


def test_oversized_image_is_rejected(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr(main, "MAX_GENERATED_BYTES", 10)
    monkeypatch.setattr(main.requests, "get", lambda *a, **k: _Resp(_png()))
    assert c.post("/generate_image", data={"prompt": "a red velvet sofa"}).status_code == 502


@pytest.mark.parametrize("prompt", ["", "ab", "x" * 301])
def test_prompt_length_is_validated(client, prompt):
    c, _ = client
    assert c.post("/generate_image", data={"prompt": prompt}).status_code == 422


def test_valid_png_with_wrong_content_type_is_rejected(client, monkeypatch):
    c, out = client
    monkeypatch.setattr(main.requests, "get", lambda *a, **k: _Resp(_png(), 200, "text/html"))
    assert c.post("/generate_image", data={"prompt": "a red velvet sofa"}).status_code == 502
    assert list(out.glob("*.jpg")) == []