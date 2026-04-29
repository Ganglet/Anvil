# Attack Engine — Multi-Strategy Adversarial Attack

**Phase:** 3 — Multi-Strategy Attack Engine
**Status:** Complete
**Date:** April 2026

---

## Objective

Build a suite of adversarial attack strategies that take a trained model and generate inputs designed to fool it. Each strategy targets the model differently. Phase 4 clusters the failures these attacks produce.

---

## What Was Built

```
attacks/
  __init__.py
  base_attack.py       — AdversarialExample dataclass + abstract BaseAttack
  fgsm.py              — Fast Gradient Sign Method
  pgd.py               — Projected Gradient Descent
  patch_attack.py      — Adversarial patch (small region optimisation)
  semantic_attack.py   — Rotation, brightness, contrast perturbations
  text_attack.py       — Synonym swap, char flip, filler word insertion
  engine.py            — Orchestrates all strategies; routes image vs. text
```

---

## AdversarialExample

Every attack returns a list of `AdversarialExample` objects — one per input sample. Each holds:

| Field | Contents |
|---|---|
| `original_input` | The clean input tensor |
| `perturbed_input` | The adversarial input tensor |
| `true_label` | Ground-truth class |
| `original_pred` | Model prediction on clean input |
| `adversarial_pred` | Model prediction on perturbed input |
| `attack_name` | String identifier of the attack |
| `attack_params` | Dict of hyperparameters used |
| `success` | `True` if `original_pred != adversarial_pred` |

---

## Attack Strategies

### FGSM (Fast Gradient Sign Method)
**Paper:** Goodfellow et al., 2014.

Single-step perturbation:
```
x_adv = x + ε · sign(∇_x L(x, y))
```
Computes the gradient of the loss w.r.t. the input, steps in the direction that increases the loss most. One backward pass per sample. Fast but shallow — a strong model resists it.

Default: ε = 0.03

---

### PGD (Projected Gradient Descent)
**Paper:** Madry et al., 2018.

Iterative FGSM with a random start inside the ε-ball. After each step, the perturbation is projected back so it never exceeds ε from the original input:
```
x_0    = x + uniform_noise(−ε, ε)
x_{t+1} = Proj_{ε}(x_t + α · sign(∇_x L(x_t, y)))
```
40 steps by default. Strictly stronger than FGSM — finds failures FGSM misses. The standard benchmark for adversarial robustness in the literature.

Default: ε = 0.03, step_size = 0.007, num_steps = 40

---

### Patch Attack
**Paper:** Brown et al., 2017.

Optimises a small square patch (8×8 pixels by default) placed at the centre of the image. The rest of the image is untouched. Gradient ascent on the patch pixels for 50 steps.

Models physical-world attacks — a sticker placed on a stop sign that fools a self-driving car's classifier. Unlike FGSM/PGD, the perturbation is not spread across the whole image; it is localised and visible.

Patch position is currently image-centre. In a full spatial saliency pipeline, Phase 2's Captum output would place it at the highest-saliency region.

Default: patch_size = 8, num_steps = 50, step_size = 0.05

---

### Semantic Attack
No gradient computation. Applies human-perceptible but "natural" transformations:
- Rotation: ±15°, ±30°
- Brightness: 0.5×, 1.5×, 0.3×, 1.8×
- Contrast: 0.5×, 1.5×, 0.3×, 2.0×

Tries each in sequence and stops at the first one that flips the prediction. Tests whether the model relies on orientation or lighting rather than the actual object.

---

### Text Attack
Token-level perturbations for DistilBERT. Decodes the input token IDs back to text, applies one of three perturbations, re-tokenises:

| Perturbation | Example |
|---|---|
| Synonym swap | "good movie" → "great movie" |
| Character flip (typo) | "acting" → "aacting" |
| Filler word insertion | "bad film" → "really bad film" |

The synonym map is small (~8 words) — sufficient for tests and demo. A production version would use WordNet or a masked-LM for context-aware substitutions.

---

## Engine

`AttackEngine` orchestrates all strategies and routes based on model type:

```python
engine = AttackEngine(model)
results = engine.run(inputs, labels, profile)
# results = {"FGSM": [...], "PGD": [...], "Patch": [...], "Semantic": [...]}

rates = engine.success_rate(results)
# rates = {"FGSM": 0.4, "PGD": 0.85, "Patch": 0.6, "Semantic": 0.3}
```

Image models get all 4 strategies. Text models get TextAttack only.

---

## Connection to Phase 2

The `profile` dict from Phase 2 is passed into every `attack()` call. Currently FGSM/PGD use it indirectly (the model they run against was profiled). PatchAttack uses `vulnerability_score` as a proxy for saliency placement. In a full pipeline, the spatial saliency map from Phase 2's Captum output would determine the exact patch coordinates.

---

## Test Results

25/25 tests passing (`tests/test_attacks.py`).

| Test group | Count | Result |
|---|---|---|
| AdversarialExample dataclass | 2 | Pass |
| FGSM | 5 | Pass |
| PGD | 3 | Pass |
| PatchAttack | 3 | Pass |
| SemanticAttack | 4 | Pass |
| TextAttack | 4 | Pass |
| AttackEngine | 4 | Pass |

Tests use `pretrained=False` (random-weight ResNet-18) and reduced step counts (PGD: 5 steps, Patch: 5 steps). They verify attack mechanics — correct shapes, bounded perturbations, correct metadata — not whether attacks succeed on a meaningful model. Real success rates emerge in Phase 8 when the full pipeline runs on a pretrained model.

---

→ See `04_failure_mode_clustering.md` after Phase 4 is complete.
