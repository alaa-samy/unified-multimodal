# Competitor bridges: UniIR and SpeechCLIP

GPU-backed `/search` services for the two academic baselines used in the thesis
experiment, each with its own ABO FAISS index, following the bridge contracts
documented in the testbed:

- UniIR (ECCV 2024): https://github.com/TIGER-AI-Lab/UniIR — text + image baseline
- SpeechCLIP (IEEE SLT 2022): https://github.com/atosystem/SpeechCLIP — voice + image baseline

Both services return stable ABO product IDs from the same 75-product fixture
catalog the testbed storefront exposes, so rankings are directly comparable.

## Layout

```
competitors/
  common/            ABO catalog loader, FAISS index helpers, ranking metrics
  uniir_bridge/      UniIR CLIP-SF model wrapper, index builder, FastAPI service, Modal app
  speechclip_bridge/ SpeechCLIP wrapper, index builder, FastAPI service, Modal app
  tests/             pipeline tests (mock encoder + real FAISS, no GPU needed)
  run_ranking_tests.py  modality-level ranking tests against deployed endpoints
  data/              abo-catalog.json candidate pool (exported from the testbed)
  results/           ranking test reports (JSON + Markdown)
```

## Separate ABO FAISS indexes

Each provider gets its own index; neither reuses the Unified SigLIP index:

| Index | Encoder | Location |
| --- | --- | --- |
| `indexes/uniir/` | fine-tuned UniIR CLIP-SF Large (ViT-L/14, 768-d) | Modal volume `uniir-abo-index` |
| `indexes/speechclip/` | SpeechCLIP checkpoint's frozen CLIP image encoder | Modal volume `speechclip-abo-index` |

Each index directory contains `abo_images.faiss` (IDMap + InnerProduct over
L2-normalized vectors, same construction as the official UniIR retriever),
`abo_ids.json` (row -> ABO product ID), and `index_meta.json` (model, dim,
catalog version, build time).

## Deploy

```bash
pip install modal
modal token new

# UniIR: builds the ABO index on GPU, then serves /search
modal run competitors/uniir_bridge/modal_app.py::build-index
modal deploy competitors/uniir_bridge/modal_app.py

# SpeechCLIP
modal run competitors/speechclip_bridge/modal_app.py::build-index
modal deploy competitors/speechclip_bridge/modal_app.py
```

Each deploy prints a `*.modal.run` URL. The UniIR checkpoint
(`clip_sf_large.pth`, 5.13 GB) and the SpeechCLIP checkpoint are downloaded
from Hugging Face into Modal volumes on first build.

## Configure the testbed endpoints

Set the environment variables the testbed adapters already read, then rebuild:

```bash
VITE_UNIIR_API_URL=https://<your-app>--uniir-abo-search-serve.modal.run
VITE_SPEECHCLIP_API_URL=https://<your-app>--speechclip-abo-search-serve.modal.run
```

Without these variables the storefront keeps the deterministic local baselines
and never claims a live model run.

## Run the real modality-level ranking tests

```bash
pip install gTTS   # optional: synthesize speech queries from ABO product names

python -m competitors.run_ranking_tests \
  --uniir-url "$VITE_UNIIR_API_URL" \
  --speechclip-url "$VITE_SPEECHCLIP_API_URL" \
  --limit 25 --top-k 10 --tts
```

Modalities tested:

| Provider | Modality | Contract fields |
| --- | --- | --- |
| UniIR | text -> image | `query_txt`, `instruction`, `candidate_modality`, `top_k` |
| UniIR | image -> image | `query_image` |
| UniIR | image+text -> image | `query_txt` + `query_image` |
| SpeechCLIP | speech -> image | `audio` |
| SpeechCLIP | speech+image -> image | `audio` + `query_image` |

Every query's ground truth is the ABO product it was derived from. Reports are
written to `competitors/results/` as JSON and Markdown with Recall@1/5/10
(UniIR hit-rate definition), MRR, and mean latency per modality.

For recorded (non-TTS) speech, drop `{item_id}.wav` files (16 kHz mono) into
`competitors/data/test-audio/` and omit `--tts`.

## Tests

```bash
pip install pytest numpy faiss-cpu
pytest competitors/tests -q
```

The tests exercise the FAISS index roundtrip, self-retrieval ranking, score
fusion, metrics, and catalog integrity with a deterministic mock encoder, so
they run on CPU without checkpoints.
