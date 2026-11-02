import numpy as np
import pytest
import torch

from embedding_backend import _normalize


def test_normalize_unwraps_nested_model_outputs():
    inner = type("NestedOutput", (), {"pooler_output": torch.tensor([[3.0, 4.0]])})()
    outer = type("BaseModelOutputWithPooling", (), {"pooler_output": inner})()
    result = _normalize(outer)
    np.testing.assert_allclose(result, [[0.6, 0.8]], rtol=1e-6, atol=1e-6)


def test_normalize_pools_last_hidden_state():
    output = type("BaseModelOutput", (), {
        "last_hidden_state": torch.tensor([[[3.0, 0.0], [5.0, 0.0]]])
    })()
    result = _normalize(output)
    np.testing.assert_allclose(result, [[1.0, 0.0]], rtol=1e-6, atol=1e-6)


def test_index_dimension_guard_reports_actual_encoder_width(monkeypatch):
    import main

    monkeypatch.setattr(main, "_index_dimension", 512)
    monkeypatch.setattr(main, "_index_checked", False)
    monkeypatch.setattr(main, "_clip_model", object())
    monkeypatch.setattr(main, "_clip_proc", object())
    monkeypatch.setattr(main, "_embedding_model_name", "google/siglip-base-patch16-224")
    monkeypatch.setattr(main, "encode_text", lambda *args, **kwargs: np.zeros((1, 768), dtype="float32"))

    with pytest.raises(RuntimeError, match="outputs 768 dimensions.*has 512"):
        main._ensure_index_compatible()
