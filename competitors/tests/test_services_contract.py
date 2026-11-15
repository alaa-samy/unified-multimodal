"""HTTP contract tests for both bridge /search endpoints.

The neural encoders are replaced with a deterministic mock so these verify the
real FastAPI multipart contract, FAISS-backed ranking, fusion, and response
shapes the testbed adapters depend on. Run with:

    pip install pytest numpy faiss-cpu fastapi python-multipart httpx pillow
    pytest competitors/tests -q
"""
from __future__ import annotations

import io
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
CATALOG_PATH = str(REPO_ROOT / "competitors" / "data" / "abo-catalog.json")

if "torch" not in sys.modules:
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        torch_stub = types.ModuleType("torch")
        torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
        sys.modules["torch"] = torch_stub

from competitors.common.faiss_index import build_index, save_index

DIM = 32


class MockCLIPSF:
    """Stands in for UniIRCLIPSF with deterministic text/image vectors."""

    def __init__(self, checkpoint_path=None, **kwargs) -> None:
        self.rng = np.random.default_rng(7)

    def _vector(self, seed: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(seed)) % (2**32))
        vector = rng.standard_normal(DIM).astype("float32")
        return vector / np.linalg.norm(vector)

    def encode_text(self, text: str, instruction=None) -> np.ndarray:
        return self._vector(f"txt:{text}")

    def encode_image(self, image) -> np.ndarray:
        return self._vector(f"img:{image.size}")

    def encode_image_batch(self, images, batch_size=32) -> np.ndarray:
        return np.stack([self._vector(f"img:{image.size}") for image in images])


class MockSpeechCLIP:
    def __init__(self, checkpoint_path=None, **kwargs) -> None:
        self.rng = np.random.default_rng(11)

    def encode_speech(self, waveforms) -> np.ndarray:
        rng = np.random.default_rng(len(waveforms[0]))
        vector = rng.standard_normal(DIM).astype("float32")
        return (vector / np.linalg.norm(vector))[None, :]

    def encode_images(self, images, batch_size=16) -> np.ndarray:
        rng = np.random.default_rng(99)
        vector = rng.standard_normal(DIM).astype("float32")
        return (vector / np.linalg.norm(vector))[None, :]


def _build_index_dir(tmp_path: Path, product_ids, vectors) -> str:
    index = build_index(np.stack(vectors).astype("float32"), product_ids)
    save_index(index, product_ids, {"provider": "mock"}, str(tmp_path))
    return str(tmp_path)


@pytest.fixture(scope="module")
def uniir_client(tmp_path_factory, monkeypatch_module_scope):
    from competitors.common.abo_catalog import load_catalog
    from competitors.uniir_bridge import model as uniir_model
    from competitors.uniir_bridge.service import create_app

    products = load_catalog()["products"]
    encoder = MockCLIPSF()
    vectors = [encoder._vector(f"img:({w}, {h})") for w, h in [(224, 224)] * len(products)]
    index_dir = _build_index_dir(tmp_path_factory.mktemp("uniir"), [p["id"] for p in products], vectors)
    monkeypatch_module_scope.setattr(uniir_model, "UniIRCLIPSF", MockCLIPSF)
    from fastapi.testclient import TestClient

    return TestClient(create_app(index_dir=index_dir, catalog_path=CATALOG_PATH)), products


@pytest.fixture(scope="module")
def speechclip_client(tmp_path_factory, monkeypatch_module_scope):
    from competitors.common.abo_catalog import load_catalog
    from competitors.speechclip_bridge import model as sc_model
    from competitors.speechclip_bridge.service import create_app

    products = load_catalog()["products"]
    encoder = MockSpeechCLIP()
    vectors = [encoder.encode_images([])[0] for _ in products]
    index_dir = _build_index_dir(tmp_path_factory.mktemp("speechclip"), [p["id"] for p in products], vectors)
    monkeypatch_module_scope.setattr(sc_model, "SpeechCLIPWrapper", MockSpeechCLIP)
    from fastapi.testclient import TestClient

    return TestClient(create_app(index_dir=index_dir, catalog_path=CATALOG_PATH)), products


@pytest.fixture(scope="module")
def monkeypatch_module_scope():
    """Module-scope monkeypatch shim (pytest's monkeypatch is function-scoped)."""

    class Scope:
        def setattr(self, target, name, value):
            setattr(target, name, value)

    return Scope()


def test_uniir_text_search_contract(uniir_client):
    client, products = uniir_client
    response = client.post(
        "/search",
        data={"query_txt": "black metal side table", "candidate_modality": "image", "top_k": "5"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "uniir"
    assert payload["total_results"] == len(payload["results"]) == 5
    for hit in payload["results"]:
        assert set(hit) == {"product_id", "score"}
        assert hit["product_id"] in {p["id"] for p in products}
        assert -2.0 <= hit["score"] <= 2.0
    scores = [hit["score"] for hit in payload["results"]]
    assert scores == sorted(scores, reverse=True)


def test_uniir_image_and_fused_search(uniir_client):
    client, _ = uniir_client
    image_bytes = io.BytesIO()
    Image.new("RGB", (224, 224), color=(10, 20, 30)).save(image_bytes, format="JPEG")
    files = {"query_image": ("q.jpg", image_bytes.getvalue(), "image/jpeg")}
    response = client.post("/search", data={"top_k": "3"}, files=files)
    assert response.status_code == 200
    assert len(response.json()["results"]) == 3
    assert response.json()["modalities"] == ["image"]

    response = client.post(
        "/search",
        data={"query_txt": "blue recliner", "top_k": "4"},
        files=files,
    )
    assert response.status_code == 200
    assert response.json()["modalities"] == ["text", "image"]


def test_uniir_rejects_empty_query(uniir_client):
    client, _ = uniir_client
    response = client.post("/search", data={"top_k": "5"})
    assert response.status_code == 400


def test_uniir_health(uniir_client):
    client, products = uniir_client
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["provider"] == "uniir"
    assert payload["index_size"] == len(products)


def test_speechclip_requires_audio(speechclip_client):
    client, _ = speechclip_client
    response = client.post("/search", data={"top_k": "5"})
    assert response.status_code == 400


def test_speechclip_speech_search_contract(speechclip_client):
    client, products = speechclip_client
    wav = _make_wav_bytes()
    response = client.post(
        "/search",
        data={"query_text": "black sofa", "candidate_modality": "image", "top_k": "6"},
        files={"audio": ("q.wav", wav, "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "speechclip"
    assert payload["modalities"] == ["speech"]
    assert payload["total_results"] == len(payload["results"]) == 6
    known_ids = {p["id"] for p in products}
    assert all(hit["product_id"] in known_ids for hit in payload["results"])


def test_speechclip_fused_speech_image_search(speechclip_client):
    client, _ = speechclip_client
    wav = _make_wav_bytes()
    image_bytes = io.BytesIO()
    Image.new("RGB", (224, 224), color=(200, 10, 10)).save(image_bytes, format="JPEG")
    response = client.post(
        "/search",
        data={"top_k": "5"},
        files={
            "audio": ("q.wav", wav, "audio/wav"),
            "query_image": ("q.jpg", image_bytes.getvalue(), "image/jpeg"),
        },
    )
    assert response.status_code == 200
    assert response.json()["modalities"] == ["speech", "image"]


def test_speechclip_health(speechclip_client):
    client, products = speechclip_client
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["provider"] == "speechclip"
    assert payload["index_size"] == len(products)


def _make_wav_bytes(seconds: float = 0.5, sr: int = 16000) -> bytes:
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        wav_file.writeframes((np.sin(np.linspace(0, 440 * seconds * 2 * np.pi, int(sr * seconds))) * 10000).astype("<i2").tobytes())
    return buffer.getvalue()
