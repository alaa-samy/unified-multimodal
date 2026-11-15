"""Modal deployment for the UniIR ABO search bridge.

Deploy:
    modal deploy competitors/uniir_bridge/modal_app.py

Build the ABO index on GPU:
    modal run competitors/uniir_bridge/modal_app.py::build-index
"""
from __future__ import annotations

import modal

app = modal.App("uniir-abo-search")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.1.0",
        "torchvision==0.16.0",
        "fastapi==0.104.1",
        "python-multipart==0.0.6",
        "Pillow==10.1.0",
        "numpy==1.24.3",
        "faiss-cpu==1.7.4",
        "huggingface_hub==0.20.3",
        "git+https://github.com/openai/CLIP.git",
    )
    .add_local_python_source("competitors")
)

checkpoint_volume = modal.Volume.from_name("uniir-checkpoints", create_if_missing=True)
index_volume = modal.Volume.from_name("uniir-abo-index", create_if_missing=True)
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
    scaledown_window=300,
)
def build_index_remote(catalog_path: str = "/cache/media/abo-catalog.json") -> dict:
    from competitors.uniir_bridge.build_index import build_uniir_abo_index
    from competitors.uniir_bridge.model import download_checkpoint

    checkpoint = download_checkpoint("/cache/checkpoints")
    checkpoint_volume.commit()
    result = build_uniir_abo_index(
        catalog_path=catalog_path,
        out_dir="/cache/index/uniir",
        checkpoint_path=checkpoint,
        image_cache_dir="/cache/media/uniir-images",
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

    from competitors.uniir_bridge.service import create_app

    os.environ.setdefault("UNIIR_INDEX_DIR", "/cache/index/uniir")
    os.environ.setdefault("UNIIR_CHECKPOINT_PATH", "/cache/checkpoints/CLIP_SF/Large/Instruct/InBatch/clip_sf_large.pth")
    os.environ.setdefault("ABO_CATALOG_PATH", "/cache/media/abo-catalog.json")
    os.environ.setdefault("UNIIR_CLIP_CACHE", "/cache/checkpoints/clip-cache")
    return create_app()


@app.local_entrypoint()
def build_index_main(catalog: str = "competitors/data/abo-catalog.json") -> None:
    import pathlib

    media_volume.put_file(catalog, "/abo-catalog.json")
    result = build_index_remote.remote(catalog_path="/cache/media/abo-catalog.json")
    print(result)
    _ = pathlib.Path(catalog)
