# Comparative Study Baselines — Master Reference Sheet

Thesis: *Unified Multimodal Architecture for Enhanced E-Commerce Product Discovery* (MULTIVIEW)
MULTIVIEW stack: Text + Image + Voice inputs · CLIP + FAISS retrieval · Whisper ASR · LLaVA VLM reasoning · Generative fallback (Pollinations) · AR viewer (model-viewer/WebXR) · Checkout · Pix3D catalog (10,069 furniture items) + Amazon dataset.

Systems are ordered from **most related** to **least related** to MULTIVIEW.

Legend — Modalities: ✅ supported · ❌ not supported. Artifacts: ✅ released · ⏳ announced/pending · ❌ not released.

---

## Ranked comparison sheet

| # | System / Paper (venue, year) | Text | Image | Voice | AR | Code / Model / Data | Paper link | Code link | Integration with my Amazon / Pix3D dataset | Role in my comparative study |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **UniECS** — Unified Multimodal E-Commerce Search Framework with Gated Cross-modal Fusion (CIKM 2025) | ✅ | ✅ | ❌ | ❌ | Code ✅ (MIT) · Models ⏳ · Benchmark M-BEER ⏳ | [arXiv:2508.13843](https://arxiv.org/abs/2508.13843) | [github.com/qzp2018/UniECS](https://github.com/qzp2018/UniECS) | Train it yourself on any paired catalog (product title + image). Amazon ESCI pairs or Pix3D both work; PyTorch + CN-CLIP, Python 3.8, needs GPU for training. Format: custom dataset dir + train.sh/eval.sh. | **Primary architectural baseline.** Same problem statement (unified text+image+combined e-commerce search). Compare MRR / nDCG@10 / R@10 across modality combinations. |
| 2 | **UniIR + M-BEIR** — Universal Multimodal Information Retrievers (ECCV 2024) | ✅ | ✅ | ❌ | ❌ | Code ✅ · Checkpoints ✅ (HF `TIGER-Lab/UniIR`) · Benchmark ✅ (1.5M queries, 5.6M candidates, incl. Fashion200k, FashionIQ) | [arXiv:2311.17136](https://arxiv.org/abs/2311.17136) | [github.com/TIGER-AI-Lab/UniIR](https://github.com/TIGER-AI-Lab/UniIR) | Zero-shot with released checkpoints. To add my data: format catalog into M-BEIR jsonl (instructions + query + candidate pool + qrels) — medium effort. UniECS benchmarks itself against UniIR, so results are cross-comparable. | **Strongest runnable unified-retrieval baseline.** Compare Recall@5/@10 for text→image, image→image, image+text→image. |
| 3 | **GME-Qwen2VL** — Improving Universal Multimodal Retrieval by Multimodal LLMs (Alibaba, 2024) | ✅ | ✅ | ❌ | ❌ | Model ✅ (HF `Alibaba-NLP/gme-Qwen2VL-2B-Instruct`) · Code ✅ | [arXiv:2412.16855](https://arxiv.org/abs/2412.16855) | [huggingface.co/Alibaba-NLP/gme-Qwen2VL-2B-Instruct](https://huggingface.co/Alibaba-NLP/gme-Qwen2VL-2B-Instruct) | **Direct integration:** encode my catalog images + titles with GME, index in my existing FAISS, rerun identical evaluation. Zero-shot, no training. 2B params — needs GPU or quantization. | **Efficiency-vs-quality baseline.** My lightweight CLIP fusion + LLaVA filters vs. a heavy MLLM unified embedder. |
| 4 | **FashionIQ / Composed Image Retrieval (CIR)** (CVPR 2021, ongoing leaderboard) | ✅ | ✅ | ❌ | ❌ | Dataset ✅ · Many baseline implementations ✅ (TIRG, CLIP4CIR, …) · Leaderboard ✅ | [CVPR 2021 paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Wu_Fashion_IQ_A_New_Dataset_Towards_Retrieving_Images_by_Natural_CVPR_2021_paper.pdf) | [Challenge site + code links](https://sites.google.com/view/cvcreative2020/fashion-iq) | Benchmark is fixed (fashion, relative captions). My Amazon data would need relative-caption annotation — not practical. Use benchmark as-is. | **Formalizes my Complex Visual Reasoning scenario** (image + modification text → product). Compare my VLM-intercept approach vs. established CIR methods on R@10/R@50. |
| 5 | **Marqo-Ecommerce-Embeddings-B/L** (2024) | ✅ | ✅ | ❌ | ❌ | Models ✅ (HF) · Eval code ✅ · Eval datasets ✅ (GoogleShopping-1m, AmazonProducts-3m) | [arXiv:2406.11230](https://arxiv.org/abs/2406.11230) | [github.com/marqo-ai/marqo-ecommerce-embeddings](https://github.com/marqo-ai/marqo-ecommerce-embeddings) | **Easiest integration:** swap `openai/clip-vit-base-patch32` → `Marqo/marqo-ecommerce-embeddings-B` (203M params, CPU-friendly) inside `_do_search` + FAISS; rerun identical Pix3D/Amazon evaluation. | **Embedding-quality baseline.** Generic CLIP vs. e-commerce-tuned encoder; published MRR/nDCG@10 numbers to anchor my results. |
| 6 | **FashionCLIP** (Cambridge + Farfetch, 2022) | ✅ | ✅ | ❌ | ❌ | Code ✅ (MIT) · Model ✅ (HF `patrickjohncyh/fashion-clip`) | [arXiv:2204.03972](https://arxiv.org/pdf/2204.03972) | [github.com/patrickjohncyh/fashion-clip](https://github.com/patrickjohncyh/fashion-clip) | Drop-in encoder alternative; Python API exposes retrieval + zero-shot classification directly. Works on any product catalog with images. | **Domain-fine-tuned CLIP baseline** (older than Marqo; cite both to show the fine-tuning lineage). |
| 7 | **Voice-Driven Multimodal Image Retrieval** — Whisper + CLIP + Pinecone + diffusion re-ranking (2025) | ❌ (speech only) | ✅ | ✅ | ❌ | Code ❌ · Data ❌ | [ResearchGate](https://www.researchgate.net/publication/404394736_A_Voice-Driven_Multimodal_Image_Retrieval_Framework_Integrating_CLIP_Embeddings_and_Diffusion-Based_Semantic_Re-ranking) | — | **Reproduce it as a baseline:** its pipeline (Whisper → CLIP → cosine vector search) is a strict subset of my stack — implementable in my backend in a day. Evaluate both on the same spoken queries. | **Closest published voice+CLIP retrieval work.** Cite as nearest prior prototype for the voice layer. |
| 8 | **AMPere + ROPE dataset** — ASR-enhanced Multimodal Product Retrieval (Kuaishou, 2024) | ✅ (ASR text) | ✅ | ✅ (transcribed) | ❌ | ROPE dataset ✅ ([github.com/adxcreative/COPE](https://github.com/adxcreative/COPE)) · Method code ❌ | [arXiv:2408.02978](https://arxiv.org/html/2408.02978v1) | [ROPE dataset](https://github.com/adxcreative/COPE) | ROPE is Kuaishou live-stream/video product data (189K products) — my Amazon data cannot replace it directly; use ROPE as-is if evaluating ASR-text product retrieval. | **Related work for voice-derived product retrieval** (shows ASR text as a modality). Cite; not a practical baseline. |
| 9 | **WebShop** (Princeton, NeurIPS 2022) | ✅ | ❌ | ❌ | ❌ | Code ✅ · Environment ✅ (1.18M products, pip-installable) · Baseline agents ✅ | [arXiv:2207.01206](https://arxiv.org/abs/2207.01206) | [github.com/princeton-nlp/WebShop](https://github.com/princeton-nlp/WebShop) | Custom catalog import possible (items.json + search-index build) — my Amazon data can be loaded with medium effort. Text-only search. | **End-to-end shopping-flow baseline** (search → select → cart → purchase). Compare task success rate and time-to-purchase. |
| 10 | **SQID** — Shopping Queries Image Dataset (SIGIR eCom Workshop 2024) | ✅ | ✅ | ❌ | ❌ | Dataset ✅ (HF `crossingminds/shopping-queries-image-dataset`) · Precomputed CLIP embeddings ✅ | [SQID paper](https://mariyahendriksen.com/data/paper_17.pdf) | [github.com/crossing-minds/shopping-queries-image-dataset](https://github.com/crossing-minds/shopping-queries-image-dataset) | **It IS an Amazon-derived benchmark** (ESCI + product images + CLIP embeddings). Use directly as external evaluation data — no integration needed. | **Ready-made Amazon benchmark** with a baseline method's embeddings to benchmark against. |
| 11 | **Amazon ESCI / Shopping Queries Dataset** (KDD Cup 2022) | ✅ | ❌ (text only; images via SQID) | ❌ | ❌ | Dataset ✅ | [amazon.science](https://www.amazon.science/code-and-datasets/shopping-queries-dataset-a-large-scale-esci-benchmark-for-improving-product-search) | [github.com/amazon-science/esci-data](https://github.com/amazon-science/esci-data) | Direct download; use as additional query/product source for my Amazon-side evaluation. | **Evaluation data source**, not a system. |
| 12 | **TS-SpeechCLIP / SpokenCOCO** — Target-Speaker Speech-Image Retrieval (2025) | ❌ (speech) | ✅ | ✅ | ❌ | Code ✅ · Datasets ✅ (SpokenCOCO, Flickr8k) | [arXiv:2509.09306](https://arxiv.org/html/2509.09306) | [github.com/Wenhao-Yang/TS-SpeechCLIP](https://github.com/Wenhao-Yang/TS-SpeechCLIP) | Generic domain (spoken captions → images), not e-commerce. My Amazon data not applicable. | **Speech→image retrieval benchmark** if I want a formal spoken-query metric (R@1/R@5) for the voice layer. |
| 13 | **AR Shopping System Through Image Search and Virtual Shop Generation** (HCI International 2020, Springer) | ✅ | ✅ | ✅ | ✅ | Code ❌ · Data ❌ | [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-030-50020-7_26) | — | No artifacts — cannot run or integrate. | **Closest published system to MULTIVIEW overall** (image search + voice commands + AR preview). Cite as nearest prior prototype; compare qualitatively in feature table. |
| 14 | **XR-Objects** (Google Research, 2024) | ✅ | ✅ | ❌ | ✅ | Open-source prototype ✅ | [Research blog](https://research.google/blog/augmented-object-intelligence-with-xr-objects/) | (open-source repo linked from blog) | Not e-commerce; no product catalog/checkout. | **Runnable AR+vision research artifact** — reference implementation for the AR interaction layer. |
| 15 | **ARShopping** — In-Store Shopping Decision Support (IEEE VIS 2022) | ✅ | ✅ | ❌ | ✅ | Code ❌ (prototype paper) | [arXiv:2207.07643](https://arxiv.org/abs/2207.07643) | — | No artifacts. | **Adopt its evaluation methodology** (user study, Likert usability scales) for my with-vs-without-AR ablation. |
| 16 | **Industry papers without artifacts** — VL-CLIP (Walmart, RecSys '25) · MOON (Alibaba, WSDM '26) · MRSE (Shopee, [arXiv:2408.14968](https://arxiv.org/abs/2408.14968)) · Beyond Text two-tower (Target, 2026) | ✅ | ✅ | partial | ❌ | Code ❌ · Models ❌ | see Literature_Review_Audit.txt | — | Not integrable — no public APIs. | **Related-work citations only**; their absence of public artifacts justifies why they are not experimental baselines. |

---

## Modality coverage of the field (the research gap)

| System | Text | Image | Voice | AR | Checkout | Public artifacts |
|---|---|---|---|---|---|---|
| **MULTIVIEW (mine)** | ✅ | ✅ | ✅ | ✅ | ✅ | thesis artifact |
| UniECS | ✅ | ✅ | ❌ | ❌ | ❌ | code only |
| UniIR | ✅ | ✅ | ❌ | ❌ | ❌ | full |
| GME-Qwen2VL | ✅ | ✅ | ❌ | ❌ | ❌ | full |
| FashionIQ/CIR | ✅ | ✅ | ❌ | ❌ | ❌ | full |
| Marqo / FashionCLIP | ✅ | ✅ | ❌ | ❌ | ❌ | full |
| Voice-driven retrieval (2025) | ✅ | ✅ | ✅ | ❌ | ❌ | none |
| AR Shopping System (2020) | ✅ | ✅ | ✅ | ✅ | ❌ | none |
| WebShop | ✅ | ❌ | ❌ | ❌ | ✅ | full |

**Conclusion:** no published system combines text + image + voice input, AR display, and checkout with public artifacts. This is the documented research gap MULTIVIEW fills.

---

## Recommended experimental plan

1. **Retrieval core (main comparison):** MULTIVIEW (CLIP + mean-fusion + LLaVA filters) vs. UniIR (zero-shot checkpoints) vs. Marqo-Ecommerce-B (drop-in encoder swap) vs. GME (zero-shot). Metrics: MRR, nDCG@10, Recall@K, response time. Datasets: my Pix3D catalog + SQID (Amazon-derived).
2. **Composed/complex queries:** my multi-image + text + VLM-reasoning flow vs. FashionIQ CIR baselines on R@10/R@50.
3. **Voice layer:** reproduce the Whisper+CLIP baseline (row 7) inside my backend; evaluate on the same spoken queries (component-level comparison).
4. **AR layer (ablation, not baseline):** within-subject user study, MULTIVIEW with vs. without AR viewer, Likert scales following ARShopping's methodology.
5. **Feature-coverage table** (above) citing rows 13–16 to substantiate the novelty claim.
