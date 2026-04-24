# Project Scaffold & Model-Agnostic Interface

**Phase:** 1 — Project Scaffold & Model-Agnostic Interface  
**Status:** Complete  
**Date:** April 2026

---

## Objective

Establish the full repository skeleton and implement the model-agnostic interface that every downstream phase depends on. Every component in Phases 2–7 (attack engine, clustering, LLM agent, patching, report) calls exactly three methods on a model: `predict()`, `get_gradients()`, and `get_activations()`. This phase defines that contract and provides two concrete implementations — ResNet-18 for images and DistilBERT for text.

---

## 1. Repository Structure

```
RedQueen_Project/
├── models/
│   ├── __init__.py
│   ├── base_model.py        ← abstract interface
│   ├── image_model.py       ← ResNet-18 wrapper
│   └── text_model.py        ← DistilBERT wrapper
├── tests/
│   ├── __init__.py
│   └── test_models.py
├── Documentation/
│   ├── problems_and_decisions.md
│   └── 01_scaffolding_interface.md   ← this file
├── audit.py                 ← CLI entry point skeleton
├── requirements.txt
├── docker-compose.yml
├── .gitignore
└── LICENSE
```

---

## 2. Model-Agnostic Interface — `models/base_model.py`

Abstract base class. Any model passed to Red Queen must subclass this and implement all three abstract methods. The ABC enforces this at instantiation — a subclass that skips any method cannot be constructed.

**Three abstract methods:**

| Method | Signature | Used by |
|--------|-----------|---------|
| `predict(inputs)` | `Tensor → Tensor (logits)` | All phases — "what does the model think?" |
| `get_gradients(inputs, target_class)` | `Tensor, int → Tensor` | Phase 3 — FGSM and PGD need ∂loss/∂input |
| `get_activations(inputs, layer_name)` | `Tensor, str → Tensor` | Phase 4 — UMAP+HDBSCAN clusters penultimate-layer activations |

**Two non-abstract helpers:**

- `get_available_layers() → List[str]` — returns all hookable layer names. Overridden in both concrete classes.
- `input_shape` and `model_name` — abstract properties that enforce each wrapper to declare what it expects.

**Why abstract and not just duck-typed:** The pipeline passes model objects between phases. An abstract class guarantees at construction time that all three methods exist and have the right signatures, rather than failing at Phase 3 runtime with an `AttributeError`.

---

## 3. ResNet-18 Wrapper — `models/image_model.py`

Wraps `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)` — pretrained on ImageNet (1000 classes).

**`predict(inputs)`**
Standard forward pass inside `torch.no_grad()`. Returns raw logits of shape `(batch, 1000)`. `no_grad()` prevents gradient tape accumulation during inference — saves memory and is faster.

**`get_gradients(inputs, target_class)`**
Clones the input, enables `requires_grad=True`, runs a forward pass, calls `.backward()` on the target class logit, returns `x.grad`. This is exactly the gradient FGSM uses: `adversarial = x + ε · sign(∇x L)`. The clone is necessary because the original input tensor may not require gradients and modifying it in-place would be unsafe.

**`get_activations(inputs, layer_name)`**
Uses PyTorch **forward hooks** — callbacks registered on a specific layer that fire during the forward pass and cache the layer's output. After the pass completes, the hook is immediately removed to prevent memory accumulation across repeated calls. The `_clear_hooks()` method handles cleanup even if an exception occurs mid-forward.

Key layers for Phase 4:
- `layer4` — final residual block, shape `(batch, 512, 7, 7)`. Captures high-level semantic features.
- `avgpool` — global average pool output, shape `(batch, 512, 1, 1)`. Collapses spatial dimensions — ideal for clustering as each sample becomes a 512-dim vector.

**Why ResNet-18 (not ResNet-50 or EfficientNet):**
Oracle Always Free VM has no GPU. ResNet-18 (~11M parameters) does a full forward pass on CPU in milliseconds. ResNet-50 is 4× heavier — attack loops run 10–40 iterations, making it prohibitively slow. ResNet-18 is also the canonical model in adversarial ML literature (Goodfellow 2014 and follow-ups), so attack results are directly comparable.

---

## 4. DistilBERT Wrapper — `models/text_model.py`

Wraps `distilbert-base-uncased-finetuned-sst-2-english` — pretrained DistilBERT fine-tuned for binary sentiment classification (positive/negative, 2 classes).

**`predict(inputs)`**
Takes token ID tensors `(batch, seq_len)`, runs through the model, returns logits `(batch, 2)`.

**`predict_from_text(texts)`**
Convenience method — tokenizes a list of strings and runs predict. Used in tests and the eventual demo.

**`get_gradients(inputs, target_class)`**
Text has one fundamental difference from images: token IDs are discrete integers, not differentiable. Gradients cannot be computed w.r.t. integers. The standard approach in text adversarial attack literature is to compute gradients w.r.t. the **token embeddings** — the continuous vectors that token IDs map to. This gives a gradient tensor of shape `(batch, seq_len, 768)`. Each value tells you how much changing a particular dimension of a particular token's embedding affects the target class score — used by Phase 3's text perturbation attack.

**`get_activations(inputs, layer_name)`**
Same forward-hook approach as ResNet. DistilBERT layers return tuples — the hook unwraps `output[0]` when needed. For Phase 4, the key layer is `distilbert.transformer.layer.5` (the final transformer block), which produces `(batch, seq_len, 768)` — the contextual representation of every token.

**Safetensors alignment fix:**
HuggingFace loads `.safetensors` weight files via memory-mapping. On Apple Silicon, the mmap'd addresses can be misaligned for Apple's Accelerate BLAS SGEMM operation (`EXC_ARM_DA_ALIGN`). After `from_pretrained()`, all parameters are cloned into heap-allocated memory: `for param in model.parameters(): param.data = param.data.clone()`. See `problems_and_decisions.md` P1 for the full diagnosis.

---

## 5. Test Suite — `tests/test_models.py`

13 tests across both models. All tests use `scope="module"` fixtures — the model is loaded once per test session, not once per test (loading DistilBERT on every test would add ~2s per test).

**ImageModel tests (7):**

| Test | What it checks |
|------|----------------|
| `test_image_predict_shape` | Output shape is `(1, 1000)` |
| `test_image_predict_no_grad_leak` | `requires_grad` is False on output |
| `test_image_gradients_shape` | Gradient shape matches input shape |
| `test_image_gradients_not_zero` | Gradients are non-zero (zero = broken backward pass) |
| `test_image_activations_layer4` | `layer4` returns 4D tensor, batch dim = 1 |
| `test_image_activations_avgpool` | `avgpool` returns exactly `(1, 512, 1, 1)` |
| `test_image_available_layers` | `layer4` and `avgpool` exist, >10 layers total |

**TextModel tests (6):**

| Test | What it checks |
|------|----------------|
| `test_text_predict_shape` | Output shape is `(1, 2)` |
| `test_text_predict_from_text` | Batch of 2 strings returns `(2, 2)` |
| `test_text_gradients_shape` | Gradient is 3D, hidden size = 768 |
| `test_text_gradients_not_zero` | Gradient norm > 0 |
| `test_text_activations_last_transformer` | Final transformer block returns 3D tensor |
| `test_text_available_layers` | Transformer layers present, >5 total |

**Test run result:**

```
13 passed in 6.27s
```

---

## 6. CLI Skeleton — `audit.py`

Entry point for the full pipeline. Phase 1 implements the argument parser only:

```bash
python audit.py --model resnet18 --input ./samples/ --output ./report.pdf --budget 100
```

Four arguments:
- `--model` — which model to audit (`resnet18` or `distilbert`)
- `--input` — path to sample inputs
- `--output` — where to write the PDF report (default `./audit_report.pdf`)
- `--budget` — number of attack samples (default 100)

Six commented-out pipeline steps are placeholders for Phases 2–7. Each phase replaces one comment with a real function call.

---

## 7. Dependencies — `requirements.txt`

Grouped by phase so the purpose of each dependency is clear. Phase 1 installs only:

```
torch>=2.1.0
torchvision>=0.16.0
transformers>=4.40.0,<5.0.0
pytest>=8.0.0
```

`transformers` is pinned `<5.0.0` — transformers 5.x is a major version bump with breaking changes in model forward signatures. The remaining groups (captum, umap-learn, langgraph, etc.) are present in the file for future phases but not yet installed in the venv.

---

## 8. `.gitignore`

Covers: `venv/`, `__pycache__/`, `.env`, generated outputs (`outputs/`, `*.pdf`), model weights (`*.pt`, `*.pth`, `*.bin`, `*.safetensors`), `.DS_Store`, IDE folders. Model weights are excluded because they belong on HuggingFace Hub, not in git. `.env` is excluded because the Groq API key will live there in Phase 5.

---

## 9. `docker-compose.yml`

Skeleton only — wires in the `GROQ_API_KEY` environment variable and mounts an `outputs/` volume. Phase 8 fills in the Dockerfile and FastAPI service.

---

## Phase 1 Completion

- [x] `models/base_model.py` — abstract interface with 3 methods
- [x] `models/image_model.py` — ResNet-18 wrapper, all 3 methods implemented
- [x] `models/text_model.py` — DistilBERT wrapper, all 3 methods implemented
- [x] `tests/test_models.py` — 13 tests, all passing
- [x] `audit.py` — CLI skeleton with argument parser
- [x] `requirements.txt` — full dependency list across all 8 phases
- [x] `docker-compose.yml` — skeleton for Phase 8
- [x] `.gitignore` — all standard Python/ML exclusions
- [x] `Documentation/problems_and_decisions.md` — P1 entry written (safetensors alignment bug)

→ See `02_attack_surface_profiler.md`
