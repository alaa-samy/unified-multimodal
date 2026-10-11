"""Model adapter for text/image embeddings used by retrieval and indexing.

The default is SigLIP, which generally gives better image-text alignment than
legacy CLIP ViT-B/32. Set EMBEDDING_MODEL=clip to keep the old baseline.
Indexes must be rebuilt whenever the model changes.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from PIL import Image


DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "google/siglip-base-patch16-224")


def load_embedding_model(device: torch.device):
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("Transformers is required. Upgrade with: python -m pip install -U 'transformers>=4.41.0'") from exc

    model_name = DEFAULT_MODEL
    if model_name == "clip":
        model_name = "openai/clip-vit-large-patch14"
    try:
        processor = AutoProcessor.from_pretrained(model_name)
    except ValueError as exc:
        if model_name.startswith("google/siglip"):
            raise RuntimeError(
                "This SigLIP model needs a newer Transformers package. "
                "Run: python -m pip install -U 'transformers>=4.41.0'"
            ) from exc
        raise
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return model, processor, model_name


def _normalize(values: torch.Tensor) -> np.ndarray:
    values = values.float()
    values = values / values.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return values.detach().cpu().numpy().astype("float32")


def encode_text(model: Any, processor: Any, text: list[str], device: torch.device) -> np.ndarray:
    kwargs = {"text": text, "return_tensors": "pt", "padding": True, "truncation": True}
    inputs = processor(**kwargs).to(device)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    return _normalize(features)


def encode_image(model: Any, processor: Any, image: Image.Image, device: torch.device) -> np.ndarray:
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return _normalize(features)
