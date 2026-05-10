---
title: Anvil
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# ANVIL — Adversarial Neural Vulnerability Inspection and Learning

**A closed-loop ML red-teaming system that attacks any neural network, discovers its failure taxonomy, explains each vulnerability class with RAG-grounded LLM reasoning, patches autonomously, and produces a professional PDF audit report — zero human decisions required.**

ANVIL is an end-to-end autonomous pipeline for adversarial robustness auditing. It treats a trained neural network as a black-box target, systematically probes its decision boundaries through gradient-informed adversarial attacks, clusters the resulting failure modes into an emergent vulnerability taxonomy using non-linear manifold learning, and delegates explanation and remediation to a stateful LLM agent grounded in published adversarial ML research. The output is a structured audit report indistinguishable in depth from a human-written red-team assessment — produced in 3–5 minutes.

---

## Architecture

```
                         ┌────────────────────┐
                         │   Target Model     │
                         │  (any PyTorch net) │
                         └────────┬───────────┘
                                  │
                                  ▼
         ┌────────────────────────────────────────────────────────┐
         │  PHASE 1 — Model-Agnostic Interface                    │
         │  BaseModel ABC · predict() · get_gradients()           │
         │  get_activations() · ResNet-18 + DistilBERT wrappers  │
         └────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
         ┌────────────────────────────────────────────────────────┐
         │  PHASE 2 — Attack Surface Profiler                     │
         │  Captum Integrated Gradients + Saliency               │
         │  Gradient norm · Activation entropy · Layer ranking   │
         │  → vulnerability score + ordered attack priority list  │
         └────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
         ┌────────────────────────────────────────────────────────┐
         │  PHASE 3 — Multi-Strategy Attack Engine                │
         │  FGSM · PGD (PyTorch autograd, not Cleverhans)        │
         │  Adversarial Patch (Brown et al. 2017) · Semantic     │
         │  → AdversarialExample objects with full metadata       │
         └────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
         ┌────────────────────────────────────────────────────────┐
         │  PHASE 4 — Failure Mode Clustering                     │
         │  Penultimate-layer activation extraction               │
         │  UMAP (n_neighbors adaptive) → HDBSCAN (auto-k)       │
         │  Fallback: PCA if N too small for UMAP                 │
         │  → VulnerabilityTaxonomy (clusters + noise label -1)  │
         └────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
         ┌────────────────────────────────────────────────────────┐
         │  PHASE 5 — LLM Explanation Agent                       │
         │  LangGraph stateful agent                              │
         │  FAISS RAG over 10 adversarial ML papers              │
         │  nomic-embed-text embeddings · Gemini 2.5 Flash       │
         │  → grounded explanation + patch strategy per cluster   │
         └────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
         ┌────────────────────────────────────────────────────────┐
         │  PHASE 6 — Autonomous Patching                         │
         │  4 strategies: adversarial training · stylized aug    │
         │  counterfactual gen · targeted augmentation           │
         │  3-retry loop with escalating strategy                 │
         │  Safety gate: 0.6×resistance_gain + 0.4×acc_ret ≥ 0.70│
         │  AND clean accuracy drop ≤ 3%                          │
         │  → PatchReport per cluster                             │
         └────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
         ┌────────────────────────────────────────────────────────┐
         │  PHASE 7 — PDF Audit Report                            │
         │  ReportLab + matplotlib radar charts                   │
         │  Cover · executive summary · per-cluster LLM analysis  │
         │  patch results · methodology appendix                  │
         │  → audit_report.pdf                                    │
         └────────────────────────┬───────────────────────────────┘
                                  │
                                  ▼
         ┌────────────────────────────────────────────────────────┐
         │  PHASE 8 — API + Deployment                            │
         │  FastAPI async job queue (BackgroundTasks)             │
         │  CORS for ganglet.github.io                            │
         │  POST /audit/upload · GET /audit/job/{id}             │
         │  Docker on HuggingFace Spaces                          │
         └────────────────────────────────────────────────────────┘
```

---

## Technical Deep-Dive

### Phase 1 — Model-Agnostic Interface

Any PyTorch model plugs into ANVIL by subclassing a single abstract base class (`BaseModel`) and implementing three methods: `predict(x)`, `get_gradients(x, target)`, and `get_activations(x, layer)`. This interface decouples the entire pipeline from any specific architecture — the attack engine, clustering module, and patching strategies operate exclusively through these three primitives. ANVIL ships with two concrete wrappers: `ResNet18Wrapper` (ImageNet, 1000 classes, torchvision pretrained) and `DistilBERTWrapper` (SST-2 sentiment, HuggingFace pretrained). Adding a third model requires subclassing and implementing the three methods — no other changes anywhere.

### Phase 2 — Attack Surface Profiler

The profiler uses Captum's `IntegratedGradients` and `Saliency` attribution methods to measure how sensitive each layer is to input perturbation. For each layer, it computes the L2 norm of the gradient of the loss with respect to the layer's output (gradient norm) and the Shannon entropy of post-ReLU activation distributions (activation entropy). The vulnerability score is a weighted combination: `score = 0.6 × mean_gradient_norm + 0.4 × mean_entropy`, producing a scalar in [0, 1] that summarizes the model's overall susceptibility to adversarial perturbation. The profiler also ranks layers by their individual vulnerability sub-scores, producing an ordered attack priority list that Phase 3 uses to target the most exploitable layers first.

### Phase 3 — Multi-Strategy Attack Engine

FGSM and PGD are implemented entirely from scratch using PyTorch autograd — no Cleverhans, no ART. FGSM computes `x_adv = x + ε × sign(∇_x L(f(x), y))` in a single forward-backward pass. PGD iterates FGSM with step size `α` for `k` steps, projecting back onto the ε-ball after each step using `torch.clamp`. Adversarial patch attacks follow Brown et al. (2017): a fixed-size patch is randomly placed on the image and optimized independently of the global perturbation budget, making it particularly threatening in physical-world settings. Semantic perturbations (brightness, contrast, rotation, color jitter) exercise the model's invariance to non-additive transformations. Every successful attack instantiates an `AdversarialExample` dataclass carrying the original image, perturbed image, true label, predicted label, attack type, epsilon, and per-sample confidence scores.

### Phase 4 — Failure Mode Clustering

Activations from the penultimate layer are extracted for every successful adversarial example — these high-dimensional feature vectors encode *why* the model was fooled, not just *that* it was fooled. UMAP reduces these to 2D using a non-linear manifold approximation; `n_neighbors` is set adaptively (min 5, max 15) based on the sample count to avoid degenerate neighborhoods on small datasets. HDBSCAN then clusters the 2D embedding without requiring a pre-specified cluster count: it identifies arbitrarily-shaped density regions and labels noisy outliers as cluster `-1`. If the sample count is too small for UMAP to produce a stable embedding (N < 20), the pipeline falls back to PCA to ensure the clustering step always completes. The result is a `VulnerabilityTaxonomy` — a structured map of the model's distinct failure modes.

### Phase 5 — LLM Explanation Agent

A LangGraph stateful agent orchestrates the explanation workflow: for each cluster in the taxonomy, it retrieves the top-k relevant chunks from a FAISS vector index built over 10 adversarial ML papers (Goodfellow et al. 2015, Madry et al. 2018, Carlini & Wagner 2017, Brown et al. 2017, and six others), using `nomic-embed-text` for dense retrieval. The retrieved chunks, cluster statistics (centroid coordinates, member attack types, confidence distributions), and cluster label are injected into a structured prompt, and Gemini 2.5 Flash generates a grounded technical explanation of why this failure mode exists and which mitigation strategy is best suited to address it. LangGraph's state graph ensures the agent can revisit earlier reasoning steps if the first explanation fails a basic coherence check.

### Phase 6 — Autonomous Patching

The patching engine implements four strategies: adversarial training (fine-tuning on the attack set with label correction), stylized augmentation (domain-randomization using style transfer to broaden the training distribution), counterfactual generation (synthesizing near-boundary examples to tighten the decision boundary), and targeted augmentation (cluster-specific oversampling). The engine selects the strategy recommended by Phase 5, applies it, and evaluates the result against a composite safety gate: `score = 0.6 × resistance_gain + 0.4 × accuracy_retention`. A patch passes only if `score ≥ 0.70` AND the drop in clean accuracy is ≤ 3%. If the first strategy fails, the engine escalates to the next in the priority list and retries — up to 3 total attempts per cluster — before logging the failure and continuing to the next cluster. This gate makes the robustness–accuracy tradeoff explicit and auditable.

### Phase 7 — PDF Audit Report

ReportLab assembles a multi-page structured document: a cover page with audit metadata, an executive summary with key statistics, a radar chart (matplotlib, exported as a PNG embedded in the PDF) mapping attack success rates across attack types, per-cluster cards containing the LLM explanation, representative adversarial examples, and patch outcome, a full methodology appendix, and a limitations section. The radar chart gives a visual signature of the model's vulnerability profile — a flat polygon indicates uniform robustness; a spiked polygon reveals asymmetric weaknesses.

### Phase 8 — FastAPI API + Docker

The API layer wraps the entire pipeline in a FastAPI application with asynchronous job management using `BackgroundTasks`. `POST /audit/upload` accepts multipart form data (image files + model name + budget), enqueues a background job, and returns a `job_id` immediately. `GET /audit/job/{job_id}` is a polling endpoint returning `{status, vulnerability_score, clusters_found, clusters_patched, report_filename}`. CORS is configured to allow requests from `ganglet.github.io`. The application is packaged in a Docker image deployed on HuggingFace Spaces with persistent report storage.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Model-agnostic ABC** | Attacking a specific architecture provides no generalizable conclusions. Decoupling the pipeline through a three-method interface means every finding is about the model's learned representations, not implementation artifacts. |
| **UMAP over PCA** | PCA preserves global linear variance; UMAP preserves local non-linear neighborhood structure. Adversarial failure modes are not linearly separable in activation space — they form curved, low-dimensional manifolds. UMAP finds these manifolds; PCA cannot. |
| **HDBSCAN over k-means** | k-means requires pre-specifying k, assumes spherical clusters, and assigns every point to a cluster. HDBSCAN discovers cluster count automatically, handles non-convex shapes, and labels noisy outliers as -1 — which is the correct semantically meaningful label for attacks that don't belong to any coherent failure mode. |
| **LangGraph over a single LLM call** | A single prompt cannot maintain state across multiple clusters, revisit earlier reasoning, or branch on coherence failures. LangGraph's state graph gives the agent explicit control flow, making the reasoning traceable and debuggable. |
| **RAG over prompt-stuffing** | The 10 reference papers total ~200k tokens — too large to fit in context and too expensive to send repeatedly. FAISS retrieval reduces each query to the 3-5 most relevant chunks, keeping cost predictable and explanations grounded in specific paper text rather than hallucinated paraphrases. |
| **Composite safety gate** | Resistance gain and accuracy retention are in tension. Optimizing for one without constraining the other produces useless patches (a model that always predicts "cat" has 100% resistance). The weighted composite with a hard accuracy cap makes the tradeoff explicit and sets a defensible minimum bar. |

---

## Quick Start

### CLI

```bash
git clone https://github.com/Ganglet/Anvil.git
cd Anvil_Project
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY=your_key_here
python audit.py --model resnet18 --input ./samples/ --output ./report.pdf --budget 20
```

### Docker

```bash
echo "GOOGLE_API_KEY=your_key_here" > .env
docker-compose up --build
```

The API is then available at `http://localhost:8000`.

---

## API Reference

| Endpoint | Method | Body / Params | Response |
|----------|--------|---------------|----------|
| `/health` | GET | — | `{"status": "ok"}` |
| `/audit/upload` | POST | `multipart/form-data`: `model` (str), `budget` (int), `files[]` (images) | `{"job_id": "uuid4"}` |
| `/audit/job/{job_id}` | GET | `job_id` path param | `{"status": "pending\|running\|complete\|error", "vulnerability_score": float, "clusters_found": int, "clusters_patched": int, "report_filename": str}` |
| `/report/{filename}` | GET | `filename` path param | Binary PDF download |

**Polling pattern:** POST to `/audit/upload`, store `job_id`, poll `GET /audit/job/{job_id}` every 6 seconds until `status` is `"complete"` or `"error"`.

---

## CLI Flags

```bash
python audit.py --model resnet18 --input ./samples/ --output ./report.pdf --budget 100
```

| Flag | Description |
|------|-------------|
| `--model` | `resnet18` or `distilbert` |
| `--input` | Directory of `.jpg`/`.png` images (resnet18) or `.txt` files (distilbert). Falls back to synthetic inputs if empty. |
| `--output` | Output PDF path (default: `./audit_report.pdf`) |
| `--budget` | Number of attack samples per strategy (default: 20) |

---

## Project Structure

```
Anvil_Project/
├── audit.py                  # CLI entrypoint — orchestrates all 8 phases
├── api.py                    # FastAPI application + job queue
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── models/
│   ├── base_model.py         # BaseModel ABC
│   ├── resnet_wrapper.py     # ResNet-18 wrapper
│   └── distilbert_wrapper.py # DistilBERT wrapper
├── phases/
│   ├── phase2_profiler.py    # Attack surface profiler (Captum)
│   ├── phase3_attacks.py     # FGSM, PGD, patch, semantic attacks
│   ├── phase4_clustering.py  # UMAP + HDBSCAN clustering
│   ├── phase5_agent.py       # LangGraph agent + FAISS RAG
│   ├── phase6_patching.py    # Autonomous patching + safety gate
│   └── phase7_report.py      # PDF generation (ReportLab + matplotlib)
├── rag/
│   └── papers/               # 10 adversarial ML papers (PDF)
├── docs/
│   └── index.html            # GitHub Pages frontend
└── Documentation/
    ├── phase1.md … phase8.md
    └── problems_and_decisions.md
```

---

## Stack

| Component | Technology |
|-----------|-----------|
| Deep learning | PyTorch |
| Adversarial attacks | PyTorch autograd — FGSM, PGD from scratch; Adversarial Patch (Brown et al. 2017) |
| Interpretability | Captum (Integrated Gradients, Saliency) |
| Dimensionality reduction | UMAP (non-linear manifold; adaptive n_neighbors) |
| Clustering | HDBSCAN (auto cluster count; noise label -1) |
| LLM agent orchestration | LangGraph (stateful graph) |
| LLM | Gemini 2.5 Flash |
| RAG vector store | FAISS |
| RAG embeddings | nomic-embed-text |
| Report generation | ReportLab + matplotlib (radar charts) |
| API | FastAPI + Uvicorn |
| Job queue | FastAPI BackgroundTasks |
| Deployment | Docker on HuggingFace Spaces |
| Frontend | GitHub Pages (vanilla JS) |

---

## Models Supported

| Model | Task | Source | Input |
|-------|------|--------|-------|
| ResNet-18 | ImageNet classification (1000 classes) | torchvision pretrained | JPEG/PNG images |
| DistilBERT | Sentiment classification (SST-2) | HuggingFace pretrained | Plain text files |

Adding a new model: subclass `BaseModel` in `models/` and implement `predict()`, `get_gradients()`, and `get_activations()`. No other changes required.

---

## Limitations

| Limitation | Interview answer |
|-----------|-----------------|
| Classification models only | This is the correct scope for a solo end-to-end system. The model-agnostic interface deliberately separates the attack surface from the architecture — extending to generative models means implementing the same three methods for a generative wrapper, with Phase 3 needing a new output-space attack metric. |
| Adversarial training can reduce clean accuracy | The composite safety gate (accuracy drop ≤ 3%) explicitly measures and enforces this tradeoff. Every patch result includes before/after accuracy so the tradeoff is auditable, not hidden. |
| LLM explanations are hypotheses, not proofs | They are grounded in specific chunks from published papers via RAG, and validated against cluster statistics (attack type distributions, centroid distances). The explanation cites its sources; it does not hallucinate mechanisms. |
| Synthetic inputs used as fallback | The `--input` flag loads real images from disk. Synthetic fallback makes the pipeline self-contained for CI/demo purposes where no dataset is available. |
| Phase 6 often fails safety gate on synthetic data | Labels are the model's own predictions on synthetic inputs, so baseline accuracy is 1.0 by construction. Any weight change risks a >3% drop. This is a dataset artifact, not a pipeline bug — it passes correctly on real labeled datasets. |
| Single-GPU, sequential execution | Each phase runs sequentially. Parallelizing the per-cluster LLM calls and per-strategy patch evaluations is a straightforward next step that would cut wall-clock time by ~60%. |

---

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Model-agnostic ABC + ResNet-18/DistilBERT wrappers | Complete |
| 2 | Attack surface profiler (Captum, gradient norm, entropy) | Complete |
| 3 | Multi-strategy attack engine (FGSM, PGD, patch, semantic) | Complete |
| 4 | Failure mode clustering (UMAP + HDBSCAN) | Complete |
| 5 | LLM explanation agent (LangGraph + FAISS RAG + Gemini) | Complete |
| 6 | Autonomous patching (4 strategies, safety gate) | Complete |
| 7 | PDF audit report (ReportLab, radar charts) | Complete |
| 8 | FastAPI async API + Docker + HuggingFace Spaces deployment | Complete |

---

## Documentation

Full build log in [`Documentation/`](Documentation/) — one markdown file per phase plus a running [`problems_and_decisions.md`](Documentation/problems_and_decisions.md) logging every non-trivial design choice made during development.
