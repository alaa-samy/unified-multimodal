"""SpeechCLIP (IEEE SLT 2022) model wrapper.

Follows the official repo (atosystem/SpeechCLIP):
- KWClip_GeneralTransformer aligns a HuBERT speech encoder with a frozen CLIP
  image space through a parallel keyword branch (avssl/model/kwClip.py).
- Speech embeddings: model.encode_speech(wav)["parallel_audio_feat"] (already
  L2-normalized), example.py.
- Image embeddings: model.clip.encode_image(tensor) with the CLIP transform
  from the checkpoint config, so speech and ABO images share one space.
- Checkpoints: https://huggingface.co/speechclip/models (Flickr8k / SpokenCOCO).
  ABO retrieval is a zero-shot transfer baseline.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

DEFAULT_REPO_DIR = os.environ.get("SPEECHCLIP_REPO", "/SpeechCLIP")
DEFAULT_CONFIG = "configs/spchclp_p_large_flickr.yaml"
HF_REPO_ID = "speechclip/models"
DEFAULT_CKPT_FILE = "large/flickr/parallel/epoch_56-step_6668-val_recall_mean_10_89.0000.ckpt"


def ensure_repo_on_path(repo_dir: Optional[str] = None) -> str:
    repo = Path(repo_dir or DEFAULT_REPO_DIR)
    if not repo.exists():
        raise FileNotFoundError(
            f"SpeechCLIP repo not found at {repo}; clone https://github.com/atosystem/SpeechCLIP there"
        )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    return str(repo)


def download_checkpoint(target_dir: str, ckpt_file: str = DEFAULT_CKPT_FILE) -> str:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=HF_REPO_ID, filename=ckpt_file, local_dir=target_dir)
    return str(path)


class SpeechCLIPWrapper:
    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        repo_dir: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        repo = ensure_repo_on_path(repo_dir)
        import torch
        import avssl.model
        import yaml

        self.torch = torch
        resolved_config = Path(config_path or (Path(__file__).parent / DEFAULT_CONFIG))
        with open(resolved_config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        clip_config = config.get("clip", {})
        reduce_path = clip_config.get("reduce_subword_embbedding")
        if reduce_path and not os.path.isabs(reduce_path) and not os.path.exists(reduce_path):
            clip_config["reduce_subword_embbedding"] = str(Path(repo) / reduce_path)
        from avssl.base import OrderedNamespace

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = avssl.model.KWClip_GeneralTransformer(OrderedNamespace(**config))
        self.model.load_from_checkpoint(checkpoint_path, map_location="cpu")
        self.model.to(self.device)
        self.model.eval()
        self.clip_transform = self.model.clip.image_preprocess

    def encode_speech(self, waveforms: List[np.ndarray]) -> np.ndarray:
        torch = self.torch
        wav_tensors = [torch.FloatTensor(wav).to(self.device) for wav in waveforms]
        with torch.inference_mode():
            output = self.model.encode_speech(wav=wav_tensors)
            features = output["parallel_audio_feat"]
            if features is None:
                raise ValueError("checkpoint has no parallel branch; use a parallel SpeechCLIP checkpoint")
        return features.float().cpu().numpy().astype("float32")

    def encode_images(self, images: List, batch_size: int = 32) -> np.ndarray:
        torch = self.torch
        transform = self.clip_transform
        chunks: List[np.ndarray] = []
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size]
            tensors = torch.stack([transform(image.convert("RGB")) for image in batch]).to(self.device)
            with torch.inference_mode():
                features = self.model.clip.encode_image(tensors)
            features = features / features.norm(dim=-1, keepdim=True)
            chunks.append(features.float().cpu().numpy().astype("float32"))
        return np.concatenate(chunks, axis=0)
