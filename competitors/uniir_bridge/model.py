"""UniIR (ECCV 2024) CLIP-SF model wrapper.

Follows the official repo (TIGER-AI-Lab/UniIR):
- CLIPScoreFusion uses a fine-tuned OpenAI CLIP (ViT-L/14 for the Large model).
- Query text is embedded as "f{instruction} {query_txt}" (mbeir_dataset.py).
- Retrieval is score-level fusion: candidate score = cos(text, cand) + cos(image, cand).
- Checkpoint: TIGER-Lab/UniIR, CLIP_SF/Large/Instruct/InBatch/clip_sf_large.pth,
  loaded with model.load_state_dict(checkpoint["model"]) (train.py).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from competitors.common.defaults import DEFAULT_INSTRUCTION

DEFAULT_CLIP_MODEL = "ViT-L/14"
HF_REPO_ID = "TIGER-Lab/UniIR"
HF_CKPT_FILE = "CLIP_SF/Large/Instruct/InBatch/clip_sf_large.pth"


def download_checkpoint(target_dir: str) -> str:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_CKPT_FILE, local_dir=target_dir)
    return str(path)


class UniIRCLIPSF:
    def __init__(
        self,
        device: Optional[str] = None,
        clip_model_name: str = DEFAULT_CLIP_MODEL,
        checkpoint_path: Optional[str] = None,
        clip_download_root: Optional[str] = None,
    ) -> None:
        import clip

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model_name = clip_model_name
        download_root = clip_download_root or os.environ.get("UNIIR_CLIP_CACHE")
        self.model, self.preprocess = clip.load(clip_model_name, device=self.device, download_root=download_root)
        if checkpoint_path:
            self._load_finetuned(checkpoint_path)
        self.model.eval()

    def _load_finetuned(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"UniIR checkpoint: {len(missing)} missing keys, {len(unexpected)} unexpected keys")

    def encode_text(self, text: str, instruction: Optional[str] = None) -> np.ndarray:
        import clip

        prompt = f"{(instruction or DEFAULT_INSTRUCTION).strip()} {text}".strip() or DEFAULT_INSTRUCTION
        tokens = clip.tokenize([prompt], truncate=True).to(self.device)
        with torch.inference_mode():
            features = self.model.encode_text(tokens)
        return self._normalize(features)[0]

    def encode_image(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            features = self.model.encode_image(tensor)
        return self._normalize(features)[0]

    def encode_image_batch(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        chunks: List[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            tensors = torch.stack([self.preprocess(image.convert("RGB")) for image in batch]).to(self.device)
            with torch.inference_mode():
                features = self.model.encode_image(tensors)
            chunks.append(self._normalize(features).cpu())
        return np.concatenate(chunks, axis=0)

    def _normalize(self, features: torch.Tensor) -> np.ndarray:
        features = features / features.norm(dim=-1, keepdim=True)
        return features.float().cpu().numpy().astype("float32")
