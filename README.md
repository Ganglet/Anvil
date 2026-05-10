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

Autonomous ML red-teaming: attack any neural network, cluster its failure modes, explain each vulnerability with RAG-grounded LLM reasoning, patch autonomously, generate a PDF audit report — zero human decisions.

**Live demo → [ganglet.github.io/Anvil](https://ganglet.github.io/Anvil)**

---

## How it works

| Phase | What happens |
|-------|-------------|
| 1 — Model Interface | BaseModel ABC wraps any PyTorch net (`predict`, `get_gradients`, `get_activations`) |
| 2 — Attack Surface Profiler | Captum Integrated Gradients + Saliency → vulnerability score + attack priority list |
| 3 — Attack Engine | FGSM · PGD · Adversarial Patch · Semantic attacks, implemented from scratch in PyTorch |
| 4 — Failure Mode Clustering | Penultimate-layer activations → UMAP → HDBSCAN → `VulnerabilityTaxonomy` |
| 5 — LLM Explanation Agent | LangGraph + FAISS RAG over 10 adversarial ML papers → Gemini 2.5 Flash explanations |
| 6 — Autonomous Patching | 4 strategies, composite safety gate: score ≥ 0.70 AND accuracy drop ≤ 3% |
| 7 — PDF Report | ReportLab: cover page, radar charts, per-cluster cards, methodology appendix |
| 8 — REST API | FastAPI + Docker on HuggingFace Spaces — `POST /audit/upload`, `GET /audit/job/{id}` |

---

## Stack

PyTorch · Captum · UMAP · HDBSCAN · LangGraph · LangChain · Gemini 2.5 Flash · FAISS · sentence-transformers · ReportLab · FastAPI · Docker · HuggingFace Spaces

---

## Quick start

```bash
git clone https://github.com/Ganglet/Anvil && cd Anvil/Anvil_Project
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```

Or run the CLI directly:

```bash
python run.py --model resnet18 --budget 50
```

---

## API

```
POST /audit/upload   multipart: files[] + model + budget → { job_id }
GET  /audit/job/{id}  → { status, vulnerability_score, clusters_found, report_filename }
GET  /report/{filename}  → PDF download
GET  /health          → { status: "ok" }
```

Deployed at `https://angshuman12-anvil.hf.space` (HuggingFace Spaces, Docker, free tier — may sleep after inactivity).
