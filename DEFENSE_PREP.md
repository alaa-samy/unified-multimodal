# Defense Prep: Project Walkthrough & Q&A

Reference notes for explaining and defending the *Unified Multimodal Architecture for Enhanced E-Commerce Product Discovery* project.

---

## How the Project Was Built (in points)

1. **Research foundation** — Started from an academic proposal, *"Unified Multimodal Architecture for Enhanced E-Commerce Product Discovery"* (MIS thesis, Alexandria University), targeting the problem that e-commerce search is text-only and fragments visual search/AR as separate features.

2. **Dataset assembly** — Pulled a Kaggle "Fashion Product Images"-style metadata schema (`styles.csv`), then pivoted to a **furniture vertical** and merged it with the **Pix3D** dataset (real 3D furniture models: images + GLB models + masks). Final catalog: `backend/data/clean_products.json` — 10,069 furniture products, each tagged with category, subcategory, color, image path, and (where available) a linked 3D GLB model path.

3. **Core multimodal engine** — Used **OpenAI CLIP** (`clip-vit-base-patch32`) to embed both text queries and product images into one shared 512-dimensional vector space, so a text query and an image query can be compared directly.

4. **Vector search** — Indexed all product embeddings with **FAISS** (`image_index.faiss`, `text_index.faiss`) for fast k-NN similarity search. An experimental alternative path was explored later using **ChromaDB + SigLIP** (`reindex_to_chroma.py`), but FAISS+CLIP stayed the production default.

5. **Multi-image query support (MIQ)** — When a user uploads several images at once, their embeddings are combined via **mean-vector aggregation + normalization** (not concatenation), producing one query vector.

6. **Voice input** — Added **Whisper (tiny)** for local speech-to-text, letting users speak search queries.

7. **VLM reasoning layer** — For complex/ambiguous queries ("a chair like this but in that rug's color"), the backend calls a local **Ollama + LLaVA** vision-language model, which returns a strict JSON attribute filter, which is then applied on top of the FAISS results.

8. **Generative fallback** — If no catalog match is close enough, the app calls **Pollinations.ai** (free hosted text-to-image API, chosen over local Stable Diffusion to avoid GPU/size overhead) to generate a reference image, which is then used *as* a new image query ("Generative Design Sandbox").

9. **AR/3D preview** — Frontend renders Pix3D-derived GLB models using **`@google/model-viewer`** (WebXR-based), giving users an in-browser AR/3D view of furniture.

10. **Backend API** — Built with **FastAPI**, exposing endpoints like `/search/intelligent`, `/generate_image`, `/checkout/cart/add/{id}`, `/checkout/order`, `/sessions/*`, `/products/{id}`, `/health`. Some Clean Architecture layering (`adapters/`, `domain/`, `use_cases/`, `frameworks/`) exists alongside the main monolithic `main.py`.

11. **Frontend** — **Next.js 15 + React 19 + TypeScript + Tailwind v4**, with a unified search bar (text/image/voice), chat-style results interface, cart/checkout/payment/success flow, and an evaluation dashboard for measuring search relevance and latency.

12. **Deployment** — Two parallel serverless paths: **Vercel** (`vercel.json`, deploying `backend/main.py` as a Python function) and **Modal.com** (`modal_app.py`, a containerized FastAPI+CLIP+FAISS service with persistent volume caching). No Docker/CI setup exists yet — flagged in the project's own docs as future work.

13. **Iterative build order** (reconstructed from docs, since git history has only 6 commits, 4 unlabeled): core CLIP+FAISS text/image search first → then four "enhancement phases" layered on: (1) multi-file input handling, (2) multi-image query aggregation, (3) LLaVA/VLM integration, (4) Pollinations generative sandbox — with AR and checkout added later still. An earlier self-assessment (`local-requirements.txt`) claimed only 50–60% methodology completion with AR/checkout/LLaVA all at 0%, before later docs claimed 100% completion — i.e., these were the last features bolted on.

---

## Defense Q&A Bank

**Q: What is the core novel contribution of this architecture?**
A: A *unified* multimodal search interface that lets a user query with text, multiple images, and voice simultaneously (not as separate features), backed by one shared CLIP embedding space, with a VLM (LLaVA) layer for compositional/comparative reasoning queries that pure vector similarity can't handle, and a generative fallback (text-to-image) when no catalog item is close enough.

**Q: Why CLIP specifically, and not a custom-trained model?**
A: CLIP already aligns image and text in one embedding space from large-scale pretraining (zero-shot), so no domain-specific fine-tuning was needed to get cross-modal (text↔image) retrieval working. Trade-off: it's not fine-tuned on furniture-specific attributes, so fine-grained style/material distinctions can be weaker — a limitation worth mentioning proactively.

**Q: Why FAISS instead of a managed vector database (Pinecone, Weaviate, Chroma)?**
A: FAISS is free, runs in-process/CPU, and needs no external service — appropriate for a research prototype with ~10k products. An alternative (ChromaDB + SigLIP, in `reindex_to_chroma.py`) was explored but FAISS+CLIP was kept as the production path — a good answer if asked "did you consider alternatives?"

**Q: How do you combine multiple image queries into one search?**
A: Each image is embedded independently with CLIP, then the resulting vectors are averaged (arithmetic mean) and re-normalized to unit length before the FAISS search — a simple but effective fusion approach versus more complex learned fusion.

**Q: Why not use a learned/attention-based fusion instead of mean pooling?**
A: Mean pooling is simple, interpretable, and needs no extra training data/model; a learned fusion module would require labeled multi-image query data that didn't exist for this dataset — a reasonable scope limitation to state directly if pressed on it.

**Q: What role does the VLM (LLaVA) play that plain CLIP+FAISS can't do?**
A: CLIP+FAISS is good at "find things visually similar to X," but poor at compositional/relational reasoning ("like this chair but in that color"). LLaVA is prompted to output strict JSON attribute filters (e.g., category, color) from the images+query, and those filters are applied on top of/refine the FAISS candidate set.

**Q: Why generate an image with Pollinations.ai rather than a local diffusion model?**
A: Avoids multi-GB model downloads and GPU dependency in a research prototype/serverless deployment context; trade-off is dependency on an external free API (latency, availability, no control over model quality) — be ready to acknowledge this as a limitation.

**Q: How is the AR/3D preview implemented?**
A: Pix3D-sourced GLB (3D mesh) files are rendered client-side using Google's `<model-viewer>` web component (`@google/model-viewer`), which itself is built on Three.js/WebXR under the hood — it supports rotation and AR placement on supported devices. Note: written docs describe a hand-built Three.js viewer with custom gesture controls — verify `ARViewer.tsx` directly so the actual implementation is described accurately, not the aspirational doc description.

**Q: Where did your dataset come from and what are its limitations?**
A: Combined a Kaggle fashion-product metadata schema (repurposed for furniture) with Pix3D's real 2D+3D furniture pairs, yielding 10,069 products. Limitation: metadata is thin (only category/subcategory/color) — no rich text descriptions or material/style tags, which caps how well text search can perform.

**Q: How do you evaluate search quality?**
A: The frontend includes an Evaluation Dashboard measuring relevance and response time; check `backend/evaluation.py` for exact metric definitions before the defense, since this is a likely deep-dive question (e.g., precision@k, latency).

**Q: Why is checkout/cart/payment part of a search-architecture thesis?**
A: To demonstrate the multimodal search integrated into a realistic end-to-end e-commerce flow (not just a search demo in isolation) — proving the architecture is deployable in a production-like context, addressing the proposal's "enhanced product discovery" framing (discovery → conversion).

**Q: What's not yet production-ready / what's future work?**
A: No Docker containerization, no CI/CD pipeline, FAISS is CPU/in-memory (won't scale past current catalog size without re-architecture), thin product metadata, and dependency on external free-tier APIs (Pollinations) for generative features.

**Q: Why two deployment paths (Vercel and Modal)?**
A: Vercel serverless has a 50MB package/lambda size limit which is tight for PyTorch+CLIP+FAISS, so Modal.com (container-based, with a persistent Volume for caching model weights and data) was explored as a more robust alternative for the heavier ML workload — a good example of iterating on infra constraints, worth mentioning if asked about deployment challenges.

---

## Notes / Open Items to Verify Before Defense

- Confirm `ARViewer.tsx` actual implementation (model-viewer vs. custom Three.js) so the description given matches the code.
- Pull exact evaluation metrics from `backend/evaluation.py`.
- Be ready to explain the discrepancy between early self-assessment (50-60% complete) and later docs (100% complete) as normal iterative progress, not inconsistency.
- `backend/.env.local` and `frontend/.env.local` contain real API keys in the working tree — confirm they're gitignored before sharing/pushing the repo.
