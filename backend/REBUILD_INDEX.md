# Rebuild the Pix3D index after changing embedding models

The FAISS index and the serving model must use the same embedding model and vector dimension. If the backend reports an embedding/index mismatch, rebuild the index rather than changing dimensions manually.

## Windows PowerShell

```powershell
cd backend
$env:EMBEDDING_MODEL = "google/siglip-base-patch16-224"
python .\preprocess_pix3d.py
```

## Windows Git Bash

```bash
cd backend
export EMBEDDING_MODEL="google/siglip-base-patch16-224"
python preprocess_pix3d.py
```

Install the required model dependencies first if needed:

```bash
python -m pip install -U "transformers>=4.41.0" sentencepiece torch faiss-cpu pillow
```

The preprocessing script writes `data/index_metadata.json`, including the model name and vector dimension. Start the API with the same `EMBEDDING_MODEL` value used during preprocessing. The backend now probes the actual encoder output once before search and returns a descriptive error before FAISS can raise a low-level `matmul` exception.
