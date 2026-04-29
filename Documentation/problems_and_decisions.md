# Problems Faced & Key Decisions — ANVIL

This document is a running log of every non-trivial problem encountered and every design decision made during the project. Updated after each implementation step. Serves as source material for the README's limitations section and interview answers.

---

## P1

**Phase:** 1 — Scaffold & Model-Agnostic Interface
**Where it surfaced:** `tests/test_models.py::test_text_predict_shape`
**Problem:** Fatal `EXC_ARM_DA_ALIGN` (SIGBUS) crash inside Apple Accelerate BLAS (`libBLAS → SGEMM`) when running DistilBERT forward pass on Apple Silicon (MacBookPro18,1, M1 Pro). ResNet-18 tests passed fine. Crash report showed the faulting address was inside a 255MB memory-mapped file region — the DistilBERT weights loaded by `safetensors`.
**Root cause:** `safetensors` memory-maps weight files directly from disk. Individual tensors within the file can start at byte offsets that are not aligned to the boundary Apple's Accelerate BLAS requires for SIMD (NEON/AMX) operations. ResNet avoids this because torchvision loads `.bin` weights via pickle, which copies data into normally-allocated (aligned) heap memory.
**Decision/Fix:** After `from_pretrained()`, clone every model parameter: `for param in model.parameters(): param.data = param.data.clone()`. This copies all weights from the mmap'd region into fresh heap-allocated memory with proper alignment. Three-line fix, no external dependencies, no re-download.
**Why not alternative:** `use_safetensors=False` would fall back to `.bin` format but requires re-downloading ~260MB. Downgrading torch didn't help — the bug is in Apple's Accelerate BLAS behavior with mmap'd memory, not in PyTorch itself.

---

## D1 — UMAP + HDBSCAN over PCA + k-means for failure mode clustering

**Phase:** 4 — Failure Mode Clustering
**Decision:** Use UMAP for dimensionality reduction and HDBSCAN for clustering instead of the more common PCA + k-means.
**Why UMAP over PCA:** PCA is a linear projection — it preserves global variance but collapses non-linear structure. Neural network activations live on a highly non-linear manifold; failures that are conceptually similar (e.g. texture-driven failures) are close in activation space but that closeness is curved, not linear. UMAP preserves local neighbourhood structure and handles this manifold geometry correctly. PCA would merge clusters that UMAP correctly separates.
**Why HDBSCAN over k-means:** k-means requires specifying the number of clusters k in advance. We don't know how many distinct vulnerability types a model has — that's exactly what we're trying to discover. HDBSCAN finds clusters based on density and automatically determines the number. It also designates noise points (failures that don't fit any cluster) rather than forcing every point into a cluster like k-means does. Noise points in this context are genuinely anomalous failures worth reporting separately.
**Tradeoff accepted:** UMAP is non-deterministic (random seed must be fixed for reproducibility). HDBSCAN is slower than k-means on large datasets. Neither matters here — failure sets are typically hundreds to low thousands of samples, not millions.

---

## D2 — Groq API over local Ollama + Mistral 7B for LLM inference

**Phase:** 5 — LLM Explanation Agent
**Decision:** Use Groq API (Llama 3 70B, free tier) instead of running Ollama + Mistral 7B locally as specified in the original blueprint.
**Why:** Mistral 7B requires a GPU for reasonable inference speed. Oracle Always Free VM has no GPU — running a 7B model on 4 ARM CPU cores would make each LangGraph node take minutes. Groq's inference hardware runs Llama 3 70B at ~500ms per call, free tier, with no local resource cost. A bigger model (70B vs 7B) also produces better explanations for adversarial ML concepts.
**Why not alternative:** HuggingFace Inference API is rate-limited and slower. OpenAI costs money. Self-hosted GPU instances cost ~$380/month on AWS.
**Code impact:** ~10 line change in `agent/nodes/` — swap `OllamaLLM` for `ChatGroq`. Architecture is otherwise identical.

---

## D3 — Local FAISS index over cloud vector DB for RAG

**Phase:** 5 — LLM Explanation Agent
**Decision:** Store adversarial ML paper embeddings in a local FAISS index file rather than a cloud vector store (Pinecone, Weaviate, etc.).
**Why:** The knowledge base is small — summaries of ~10 adversarial ML papers. A local FAISS index handles this in milliseconds with zero network latency and zero cost. Cloud vector DBs are designed for millions of vectors. Using Pinecone here would be over-engineering and introduce an external dependency with an API key.
**How it works:** Paper summaries are embedded once using `nomic-embed-text`, stored as a FAISS flat index file under `knowledge_base/faiss_index/`. LlamaIndex manages the embed-and-retrieve pipeline. The index file is committed to git (it's small, ~2MB).

---

## D4 — ResNet-18 over larger vision models as the image test subject

**Phase:** 1 — Scaffold, applies to Phases 2–6
**Decision:** Use ResNet-18 (pretrained, ImageNet) as the image model, not ResNet-50, EfficientNet, or a ViT.
**Why:** Oracle Always Free VM has no GPU. ResNet-18 (~11M parameters) completes a forward pass on CPU in milliseconds. Attack loops (FGSM/PGD) run 10–40 iterations — a heavier model would make Phase 3 prohibitively slow. ResNet-18 is also the canonical benchmark model in adversarial ML literature (used in Goodfellow 2014 and most follow-up work), so attack results are directly comparable.
**Why not ViT:** Vision Transformers have a qualitatively different attack surface (attention mechanism vs convolutions). Interesting for future work but adds complexity without justification for Phase 1.

---

## D5 — Abstract base class over duck typing for model interface

**Phase:** 1 — Scaffold
**Decision:** `BaseModel` is an ABC with `@abstractmethod` decorators rather than an informal protocol/duck-typed interface.
**Why:** The pipeline passes model objects between 6 phases. If a new model wrapper is added and accidentally omits `get_gradients()`, the failure would surface at Phase 3 runtime with a cryptic `AttributeError`. The ABC enforces the contract at construction time — you cannot instantiate an incomplete wrapper. For a multi-phase pipeline where each phase assumes the previous produced valid output, this early failure is significantly safer.

---

## D6 — Phase 5 recommendations select and parameterise Phase 6 strategies; not arbitrary patching

**Phase:** 6 — Autonomous Patching
**Decision:** Phase 6 has 4 pre-defined patching strategies (targeted augmentation, counterfactual generation, adversarial training, stylized augmentation). Phase 5 recommendations do not invent new strategies — they select which strategy applies to each cluster and configure its parameters.
**Why pre-defined strategies:** The space of possible patches is not open-ended. The adversarial ML literature converges on these four as the principal remediation approaches for known vulnerability classes. Pre-defining them makes Phase 6 implementable and auditable. A fully open-ended "LLM writes the patching code" approach would be unreliable.
**How Phase 5 connects to Phase 6:** Phase 5 maps each cluster to a strategy. "Texture bias" → stylized augmentation. "Background dependency" → counterfactual generation. "Gradient sensitivity" → adversarial training. "Edge-case underrepresentation" → targeted augmentation. Phase 5 also sets parameters: which layers to target, augmentation strength, number of steps. Phase 6 executes those instructions.
**Why not pre-train with all 4 strategies upfront:** Each strategy has a robustness-accuracy tradeoff. Applying all four to every model regardless of its actual vulnerabilities would unnecessarily hurt clean accuracy. Targeted patching — applying only the strategy matched to the identified vulnerability — minimises accuracy degradation.

---

## D7 — Safety gate checks both attack resistance AND clean accuracy

**Phase:** 6 — Autonomous Patching
**Decision:** The safety score ≥ 0.7 gate is not purely "does the model resist the original attacks better." It is a composite: attack resistance improvement + clean accuracy must not drop more than 3% from baseline.
**Why:** All 4 patching strategies risk overfitting — adversarial training is the classic example (the robustness-accuracy tradeoff is well-documented). A patch that makes the model immune to attacks but degrades clean accuracy from 92% to 75% is not acceptable. The composite gate catches this. If clean accuracy drops >3%, the retry loop tries a more conservative patch (lower augmentation strength, fewer fine-tuning steps). After 3 retries the cluster is marked as unresolved and reported honestly in the Phase 7 PDF.
**Why 3%:** Chosen as a reasonable tolerance for production models. Tighter would make patching nearly impossible; looser would hide real accuracy regressions.

---

## D8 — 3 retries for the patching loop, not more or less

**Phase:** 6 — Autonomous Patching
**Decision:** The retry loop attempts patching a maximum of 3 times per cluster before marking it unresolved.
**Why 3:** Each retry applies a progressively more conservative patch — aggressive → moderate → conservative (lower augmentation strength, fewer fine-tuning steps). Three retries naturally covers this full spectrum of aggression. A 4th retry would be so diluted it provides no meaningful fix — at that point the patch is essentially doing nothing, and accepting it would be misleading. If 3 increasingly conservative attempts all fail the composite safety gate, the vulnerability represents a fundamental robustness-accuracy tradeoff that cannot be resolved without architectural changes or full retraining — both outside the scope of targeted patching.
**Practical constraint:** Oracle Always Free VM has no GPU. Each retry involves CPU fine-tuning, which takes several minutes per attempt. With multiple clusters, 3 retries each is already a significant compute budget. More retries would make the patching phase impractically slow for a demo pipeline.
**Why not 2:** Two retries only covers aggressive → conservative, skipping the moderate middle ground where many patches succeed.

---

## D9 — ~10 curated paper summaries in the RAG knowledge base, not more

**Phase:** 5 — LLM Explanation Agent
**Decision:** The FAISS knowledge base contains summaries of ~10 adversarial ML papers rather than a larger corpus.
**Why sufficient:** The 10 papers chosen (Goodfellow 2014, Szegedy 2013, Madry 2018, Geirhos 2019/2020, Ilyas 2019, Zech 2018, Carlini & Wagner 2017, Brown 2017, Papernot 2016) collectively cover every major vulnerability class (texture bias, shortcut learning, gradient sensitivity, spurious correlations, patch robustness) and every principal remediation approach. There is no major conceptual gap left uncovered.
**Why not more:** RAG retrieves the top-k most similar chunks and sends them to the LLM as context. A larger corpus with loosely related papers increases the chance that top-k slots are occupied by tangentially relevant chunks, diluting the context quality. More papers also means more curation effort — we use structured summaries optimised for retrieval, not raw PDFs. Adding papers that don't introduce new concepts wastes curation time without improving retrieval quality.
**Not a hard limit:** The FAISS index handles hundreds of papers at the same millisecond-level search speed. If a new vulnerability class emerges that isn't covered, adding a paper is trivial.

---

## D10 — Small hardcoded synonym map for TextAttack, not WordNet

**Phase:** 3 — Attack Engine
**Decision:** `text_attack.py` uses a hardcoded dict of ~8 word→synonyms rather than NLTK WordNet or a masked-LM for synonym generation.
**Why:** WordNet adds a dependency and startup cost for a feature that is not the core contribution of ANVIL. The synonym map covers enough vocabulary to demonstrate the perturbation logic and pass tests. In a production version targeting real NLP classifiers, WordNet or a context-aware masked-LM swap would be appropriate.
**Tradeoff accepted:** The current map will fail to perturb inputs that contain none of its 8 keywords. The `_char_flip` and `_word_insert` fallbacks ensure the attack still runs — they just produce weaker perturbations.
