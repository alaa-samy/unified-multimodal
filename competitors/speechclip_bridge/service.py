"""SpeechCLIP /search bridge service.

Implements the exact multipart contract documented in the testbed adapter
(client/src/providers/speechclip/README.md):

    POST /search  fields: audio, query_image, query_text, candidate_modality, top_k
    -> {"results": [{"product_id": "B07TMH6289", "score": 0.91}], "total_results": N}

Ranking runs speech-to-image retrieval over the SpeechCLIP-specific ABO FAISS
index; an optional reference image is fused by summing cosines, matching the
voice-plus-image query the testbed composer forwards.
"""
from __future__ import annotations

import io
import os
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from competitors.common.abo_catalog import catalog_version, load_catalog
from competitors.common.faiss_index import AboIndex, fuse_ranked_candidates, load_index, search_index

MAX_TOP_K = 50
FUSION_CANDIDATES = 50
TARGET_SR = 16000


def create_app(
    index_dir: Optional[str] = None,
    catalog_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> FastAPI:
    api = FastAPI(title="SpeechCLIP ABO Search Bridge")
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    state: Dict[str, Any] = {"model": None, "index": None, "catalog": None, "error": None}

    def _initialize() -> None:
        if state["index"] is not None:
            return
        resolved_index_dir = index_dir or os.environ.get("SPEECHCLIP_INDEX_DIR", "indexes/speechclip")
        resolved_catalog = catalog_path or os.environ.get("ABO_CATALOG_PATH", "competitors/data/abo-catalog.json")
        resolved_ckpt = checkpoint_path or os.environ.get("SPEECHCLIP_CHECKPOINT_PATH")
        from competitors.speechclip_bridge.model import SpeechCLIPWrapper

        state["catalog"] = load_catalog(resolved_catalog)
        state["index"] = load_index(resolved_index_dir)
        state["model"] = SpeechCLIPWrapper(checkpoint_path=resolved_ckpt)
        print(f"SpeechCLIP bridge ready: {state['index'].size} ABO vectors, dim {state['index'].dim}")

    def _get_model() -> SpeechCLIPWrapper:
        if state["model"] is None:
            _initialize()
        return state["model"]

    def _get_index() -> AboIndex:
        if state["index"] is None:
            _initialize()
        return state["index"]

    def _load_wav(data: bytes) -> np.ndarray:
        try:
            import wave

            with wave.open(io.BytesIO(data), "rb") as wav_file:
                sr = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
            dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sample_width)
            if dtype is None:
                raise ValueError(f"unsupported sample width {sample_width}")
            samples = np.frombuffer(frames, dtype=dtype).astype("float32")
            if channels > 1:
                samples = samples.reshape(-1, channels).mean(axis=1)
            if sr != TARGET_SR:
                import librosa

                samples = librosa.resample(samples, orig_sr=sr, target_sr=TARGET_SR)
            return samples
        except Exception:
            import librosa

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            try:
                wav, _ = librosa.load(tmp_path, sr=TARGET_SR, mono=True)
                return wav
            finally:
                os.unlink(tmp_path)

    @api.get("/")
    def root() -> Dict[str, Any]:
        return {"service": "speechclip-abo-search", "status": "ok" if state["index"] is not None else "cold"}

    @api.get("/health")
    def health() -> Dict[str, Any]:
        if state["index"] is None and state["error"] is None:
            try:
                _initialize()
            except Exception as exc:
                state["error"] = str(exc)
        return {
            "status": "ok" if state["index"] is not None else "error",
            "provider": "speechclip",
            "model": "SpeechCLIP parallel large (Flickr8k)",
            "index_size": state["index"].size if state["index"] else 0,
            "catalog_version": catalog_version(state["catalog"]) if state["catalog"] else None,
            "error": state["error"],
        }

    @api.post("/search")
    async def search(
        audio: Optional[UploadFile] = File(None),
        query_image: Optional[UploadFile] = File(None),
        query_text: str = Form(""),
        candidate_modality: str = Form("image"),
        top_k: int = Form(20),
    ) -> JSONResponse:
        try:
            _get_model()
            abo = _get_index()
            top_k_value = max(1, min(int(top_k), MAX_TOP_K))

            audio_bytes = await audio.read() if audio is not None else None
            image_bytes = await query_image.read() if query_image is not None else None
            if not audio_bytes:
                return JSONResponse(status_code=400, content={"error": "audio is required for SpeechCLIP"})

            model = state["model"]
            wav = _load_wav(audio_bytes)
            if len(wav) == 0:
                return JSONResponse(status_code=400, content={"error": "audio file is empty or unreadable"})

            modality_results = []
            speech_vector = model.encode_speech([wav])[0]
            modality_results.append(search_index(abo.index, speech_vector, FUSION_CANDIDATES))
            if image_bytes:
                image_vector = model.encode_images([Image.open(io.BytesIO(image_bytes))])[0]
                modality_results.append(search_index(abo.index, image_vector, FUSION_CANDIDATES))

            results = fuse_ranked_candidates(modality_results, abo.product_ids, top_k_value)[:top_k_value]
            return JSONResponse(
                content={
                    "results": results,
                    "total_results": len(results),
                    "query": query_text or "",
                    "provider": "speechclip",
                    "catalog_version": catalog_version(state["catalog"]),
                    "modalities": ["speech"] + (["image"] if image_bytes else []),
                    "candidate_modality": candidate_modality,
                }
            )
        except FileNotFoundError as exc:
            return JSONResponse(status_code=503, content={"error": str(exc)})
        except Exception as exc:
            return JSONResponse(status_code=500, content={"error": str(exc)})

    return api
