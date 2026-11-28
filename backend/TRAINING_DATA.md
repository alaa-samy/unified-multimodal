# Multimodal Training & Evaluation Data Protocol

Maps the proposal's data commitment (§7.1.3, "Data Collection and
Management") onto what the system actually uses. Read together with the
thesis Chapter 3 data sections.

## 1. Source dataset — Pix3D (CC BY 4.0)

Pix3D supplies, per item: a product photo, a category label, a textured 3D
model (.obj → converted to .glb), and an aligned image/model pair. The
catalog built by `preprocess_pix3d.py` keeps one row per valid item with:

| Proposal requirement            | How it is satisfied                                                     |
| ------------------------------- | ----------------------------------------------------------------------- |
| product images                  | `data/images/<category>/<id>.jpg` (JPEG q95)                             |
| image–text paired data          | image embedding ↔ product-name text ("Black Sofa") from the same row     |
| visual attribute annotations    | `baseColour` via CLIP zero-shot classification (`clip_color.py`, 14 colors) |
| product category hierarchies    | `masterCategory` (Furniture) → `subCategory` (sofa, chair, table, desk, bed, bookcase, wardrobe) |
| 3D/AR assets                    | `glb_path` linked from the Pix3D model field                             |

## 2. Preprocessing pipeline (proposal §7.1.3)

1. **Image preprocessing** — RGB conversion, JPEG re-encode; CLIP ViT-B/32
   preprocessing (resize 224, normalise) happens inside `encode_image`.
2. **Text encoding** — product name → CLIP text embedding (`encode_text`);
   at serving time user queries are embedded the same way, so query and
   catalog live in one shared space.
3. **Attribute annotation** — CLIP zero-shot color classification per photo
   (prompted with the category), folded into 13 canonical color families by
   `search_core.py`.
4. **Indexing** — FAISS `IndexFlatIP` over L2-normalised image embeddings
   (`image_index.faiss`). `text_index.faiss` is **evaluation-only** (the
   Text→Image retrieval protocol in `evaluation.py`); serving never reads it.
5. **Metadata augmentation** — `recompute_metadata.py` adds per-product
   color histograms (mask-based) used for the soft color boost.

## 3. Training vs. zero-shot

The system is **zero-shot**: no model weights are trained on Pix3D. CLIP's
pre-training is reused directly, which keeps the pipeline reproducible on a
free budget (no GPU training). This is a deliberate scope decision — see
"Future work" in the thesis (fine-tuned embeddings, VLM-assisted labeling).

## 4. Protocol for future fine-tuning (proposal §5 limitation)

When GPU budget exists, extend in this order:

1. **Attribute labels** — replace CLIP zero-shot colors with human-verified
   labels (color, material, pattern) for all catalog images; store as JSON
   sidecar next to `clean_products.json`.
2. **Fine-tune CLIP** — contrastive fine-tuning of CLIP on the image–name
   pairs with attribute-balanced sampling; keep ViT-B/32 for comparability.
3. **VLM-assisted data** — run the optional VLM layer (`vlm_understanding.py`,
   LLaVA via Ollama) over all catalog photos to auto-generate rich captions;
   human-review a sample before use as training text.
4. **Re-evaluate** — re-run `evaluation.py` Text→Image P@1/5/10 plus the
   multimodal user-study protocol, and compare against the zero-shot baseline
   reported in the thesis.

## 5. Optional VLM layer (proposal §7.2.1)

- `VLM_PROVIDER=ollama` + `ollama pull llava:7b` → photo-only searches are
  described in words and the description is fused with the image embedding.
- `VLM_PROVIDER=cogvlm` → local CogVLM adapter (GPU, `trust_remote_code`).
- Default: disabled. All searches behave deterministically; every VLM-assisted
  search is logged in the API response (`vlm_understanding` field) so the
  evaluation can separate VLM-assisted from pure-CLIP runs.
