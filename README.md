# ANVIL
### Adversarial Neural Vulnerability Inspection and Learning

ANVIL is an autonomous ML red-teaming agent. Point it at any trained neural network and it attacks the model with multiple adversarial strategies, clusters the failure modes, uses a RAG-grounded LLM agent to explain each vulnerability class, applies targeted patches to improve robustness, and generates a professional PDF audit report — end to end, with no human in the loop.

---

## Architecture

```
Target Model
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  Phase 2 — Attack Surface Profiler                  │
│  Gradient norms · Activation entropy · Saliency     │
│  → vulnerability score + layer attack priority      │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  Phase 3 — Multi-Strategy Attack Engine             │
│  FGSM · PGD · Adversarial Patch · Semantic         │
│  → AdversarialExample set with metadata             │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  Phase 4 — Failure Mode Clustering                  │
│  Activation extraction · UMAP · HDBSCAN            │
│  → VulnerabilityTaxonomy                            │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  Phase 5 — LLM Explanation Agent                    │
│  LangGraph · FAISS RAG · Gemini 2.5 Flash          │
│  → grounded explanation + patch strategy per cluster│
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  Phase 6 — Autonomous Patching                      │
│  4 strategies · 3-retry loop · composite safety gate│
│  → PatchReport (score ≥ 0.70 to pass)              │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│  Phase 7 — PDF Audit Report                         │
│  ReportLab · radar charts · full pipeline output    │
│  → audit_report.pdf                                 │
└─────────────────────────────────────────────────────┘
```

---

## Quick start

```bash
git clone https://github.com/Ganglet/Anvil.git
cd Anvil_Project
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY=your_key_here
python audit.py --model resnet18 --input ./samples/ --output ./report.pdf --budget 20
```

---

## Docker

```bash
echo "GOOGLE_API_KEY=your_key_here" > .env
docker-compose up --build
```

The API is then available at `http://localhost:8000`.

```bash
# Trigger an audit
curl -X POST http://localhost:8000/audit \
  -H "Content-Type: application/json" \
  -d '{"model": "resnet18", "budget": 20}'

# Download the generated report
curl http://localhost:8000/report/<filename> -o report.pdf

# Health check
curl http://localhost:8000/health
```

---

## CLI usage

```bash
python audit.py --model resnet18 --input ./samples/ --output ./report.pdf --budget 100
```

| Flag | Description |
|------|-------------|
| `--model` | `resnet18` or `distilbert` |
| `--input` | Directory of `.jpg`/`.png` images (resnet18) or `.txt` files (distilbert). Falls back to synthetic inputs if empty. |
| `--output` | Output PDF path (default: `./audit_report.pdf`) |
| `--budget` | Number of attack samples (default: 20) |

---

## Models supported

| Model | Task | Source |
|-------|------|--------|
| ResNet-18 | ImageNet classification (1000 classes) | torchvision pretrained |
| DistilBERT | Sentiment classification (SST-2) | HuggingFace pretrained |

Any model can be added by subclassing `BaseModel` and implementing three methods: `predict()`, `get_gradients()`, `get_activations()`.

---

## Stack

| Component | Technology |
|-----------|-----------|
| Deep learning | PyTorch |
| Adversarial attacks | PyTorch autograd — FGSM, PGD implemented from scratch |
| Interpretability | Captum (Integrated Gradients, saliency) |
| Clustering | UMAP + HDBSCAN |
| LLM agent | LangGraph + Gemini 2.5 Flash |
| RAG knowledge base | LlamaIndex + FAISS + nomic-embed-text |
| Report generation | ReportLab + matplotlib (radar charts) |
| API | FastAPI + Uvicorn |
| Deployment | Oracle Cloud Always Free VM + GitHub Pages |

---

## Project status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Scaffold + model-agnostic interface | ✅ Complete |
| 2 | Attack surface profiler | ✅ Complete |
| 3 | Multi-strategy attack engine | ✅ Complete |
| 4 | Failure mode clustering | ✅ Complete |
| 5 | LLM explanation agent + RAG | ✅ Complete |
| 6 | Autonomous patching | ✅ Complete |
| 7 | PDF report generation | ✅ Complete |
| 8 | Docker · FastAPI · Deployment | ✅ Complete |

---

## Documentation

Full build log in [`Documentation/`](Documentation/) — one markdown file per phase, plus a running [`problems_and_decisions.md`](Documentation/problems_and_decisions.md) logging every non-trivial design choice made during development.

---

## Limitations

| Limitation | Interview answer |
|-----------|-----------------|
| Classification models only | Correct scope for a solo project. Extending to generative models is a natural next step. |
| Adversarial training can reduce clean accuracy | My system measures and reports this tradeoff explicitly via the patch safety score. |
| LLM explanations are hypotheses, not proofs | Grounded in published research via RAG and validated against cluster statistics. |
| Synthetic inputs used by default | The `--input` flag loads real images/text from disk. Synthetic fallback makes the pipeline self-contained for demo purposes. |
| Phase 6 often fails safety gate on synthetic data | Labels equal the model's own predictions, so baseline accuracy is 1.0. Any weight change triggers the >3% accuracy drop threshold. Passes correctly on real datasets. |
