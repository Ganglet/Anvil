# ANVIL
### Adversarial Neural Vulnerability Inspection and Learning

ANVIL is an autonomous ML red-teaming agent. Point it at any trained neural network, and it attacks the model with multiple adversarial strategies, clusters the failure modes, uses a RAG-grounded LLM agent to explain each vulnerability class, applies targeted patches to improve robustness, and generates a professional PDF audit report — end to end, with no human in the loop.

---

## What it does

```
Sample inputs
    ↓
Phase 2 — Profile attack surface (gradient norms, entropy, saliency → JSON)
    ↓
Phase 3 — Multi-strategy attack engine (FGSM, PGD, patch, semantic, text attacks)
    ↓
Phase 4 — Cluster failure modes (UMAP + HDBSCAN → VulnerabilityTaxonomy)
    ↓
Phase 5 — LLM explanation agent (LangGraph + RAG over adversarial ML papers)
    ↓
Phase 6 — Autonomous patching + regression validation (safety score ≥ 0.7)
    ↓
Phase 7 — PDF audit report (ReportLab + Jinja2)
    ↓
Phase 8 — FastAPI on Oracle Cloud Always Free VM
```

---

## Models supported

| Model | Task | Source |
|-------|------|--------|
| ResNet-18 | ImageNet classification (1000 classes) | torchvision pretrained |
| DistilBERT | Sentiment classification (SST-2) | HuggingFace pretrained |

Any model can be added by subclassing `BaseModel` and implementing three methods: `predict()`, `get_gradients()`, `get_activations()`.

---

## Stack

| Component | Tool |
|-----------|------|
| Attack engine | PyTorch autograd (FGSM, PGD from scratch) |
| Saliency / attribution | Captum |
| Clustering | UMAP + HDBSCAN |
| LLM agent | LangGraph + Groq API (Llama 3 70B) |
| RAG | LlamaIndex + FAISS + nomic-embed-text |
| Report | ReportLab + Jinja2 |
| API | FastAPI + Uvicorn |
| Deployment | Oracle Cloud Always Free VM + GitHub Pages |

---

## Usage

```bash
python audit.py --model resnet18 --input ./samples/ --output ./report.pdf --budget 100
```

| Flag | Description |
|------|-------------|
| `--model` | `resnet18` or `distilbert` |
| `--input` | Path to sample inputs directory |
| `--output` | Output PDF path (default: `./audit_report.pdf`) |
| `--budget` | Number of attack samples to generate (default: 100) |

---

## Setup

```bash
git clone https://github.com/Ganglet/Anvil.git
cd Anvil_Project
python3 -m venv venv && source venv/bin/activate
pip install torch torchvision transformers captum pytest
```

> Full dependency install (all 8 phases): `pip install -r requirements.txt`

---

## Project status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Scaffold + model-agnostic interface | ✅ Complete |
| 2 | Attack surface profiler | ✅ Complete |
| 3 | Multi-strategy attack engine | 🔄 In progress |
| 4 | Failure mode clustering | ⏳ Pending |
| 5 | LLM explanation agent + RAG | ⏳ Pending |
| 6 | Autonomous patching | ⏳ Pending |
| 7 | PDF report generation | ⏳ Pending |
| 8 | Deployment | ⏳ Pending |

---

## Documentation

Full build log in [`Documentation/`](Documentation/) — one file per completed phase, plus a running [`problems_and_decisions.md`](Documentation/problems_and_decisions.md) logging every non-trivial design choice made during development.
