"""Modal deployment for the SpeechCLIP ABO search bridge.

Deploy:
    modal deploy competitors/speechclip_bridge/modal_app.py

Build the ABO index on GPU:
    modal run competitors/speechclip_bridge/modal_app.py::build-index
"""
from __future__ import annotations

import modal

app = modal.App("speechclip-abo-search")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "torch==1.13.1",
        "torchvision==0.14.1",
        "pytorch_lightning==1.5.10",
        "fairseq==0.12.2",
        "s3prl==0.3.15",
        "ftfy",
        "regex",
        "librosa==0.10.1",
        "soundfile",
        "fastapi==0.104.1",
        "python-multipart==0.0.6",
        "Pillow==10.1.0",
        "numpy==1.24.3",
        "faiss-cpu==1.7.4",
        "huggingface_hub==0.20.3",
        "PyYAML",
        "git+https://github.com/openai/CLIP.git",
    )
    .run_commands("git clone https://github.com/atosystem/SpeechCLIP /SpeechCLIP")
    .add_local_python_source("competitors")
)

checkpoint_volume = modal.Volume.from_name("speechclip-checkpoints", create_if_missing=True)
index_volume = modal.Volume.from_name("speechclip-abo-index", create_if_missing=True)
media_volume = modal.Volume.from_name("abo-media", create_if_missing=True)

GPU = "A10G"


@app.function(
    image=image,
    gpu=GPU,
    volumes={
        "/cache/checkpoints": checkpoint_volume,
        "/cache/index": index_volume,
        "/cache/media": media_volume,
    },
    timeout=3600,
    max_containers=1,
)
def build_index_remote(catalog_path: str = "/cache/media/abo-catalog.json") -> dict:
    from competitors.speechclip_bridge.build_index import build_speechclip_abo_index
    from competitors.speechclip_bridge.model import download_checkpoint

    checkpoint = download_checkpoint("/cache/checkpoints")
    checkpoint_volume.commit()
    result = build_speechclip_abo_index(
        catalog_path=catalog_path,
        out_dir="/cache/index/speechclip",
        checkpoint_path=checkpoint,
        image_cache_dir="/cache/media/speechclip-images",
    )
    index_volume.commit()
    media_volume.commit()
    return result


@app.function(
    image=image,
    gpu=GPU,
    volumes={
        "/cache/checkpoints": checkpoint_volume,
        "/cache/index": index_volume,
        "/cache/media": media_volume,
    },
    timeout=900,
    min_containers=0,
    max_containers=2,
    scaledown_window=120,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def serve():
    import os

    from competitors.speechclip_bridge.service import create_app

    os.environ.setdefault("SPEECHCLIP_INDEX_DIR", "/cache/index/speechclip")
    os.environ.setdefault("SPEECHCLIP_CHECKPOINT_PATH", "/cache/checkpoints/large/flickr/parallel/epoch_56-step_6668-val_recall_mean_10_89.0000.ckpt")
    os.environ.setdefault("SPEECHCLIP_REPO", "/SpeechCLIP")
    os.environ.setdefault("ABO_CATALOG_PATH", "/cache/media/abo-catalog.json")
    return create_app()


@app.local_entrypoint()
def build_index_main(catalog: str = "competitors/data/abo-catalog.json") -> None:
    media_volume.put_file(catalog, "/abo-catalog.json")
    result = build_index_remote.remote(catalog_path="/cache/media/abo-catalog.json")
    print(result)
