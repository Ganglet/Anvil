# Phase 6 — Autonomous Patching

## What this phase does

Phase 5 produced an `ExplanationReport` — one `ClusterExplanation` per vulnerability cluster, each specifying a patch strategy and parameters. Phase 6 executes those recommendations.

For each cluster, a retry loop:
1. Applies the recommended patch strategy to the model (fine-tuning on augmented data)
2. Evaluates a composite safety gate — did attack resistance improve AND did clean accuracy hold?
3. If the gate fails, retries with a more conservative patch (up to 3 attempts)
4. If all retries fail, marks the cluster as unresolved

Output: a `PatchReport` — one `PatchResult` per cluster, containing the strategy used, safety score, pass/fail, number of retries, resistance gain, and accuracy drop. Phase 7 renders this into the PDF audit report.

---

## Pipeline

```
ExplanationReport (Phase 5)
        │
        ▼  for each ClusterExplanation:
┌─────────────────────────────────────────────┐
│  Patcher._patch_cluster                     │
│                                             │
│  save original model weights                │
│  measure baseline clean accuracy            │
│                                             │
│  for attempt in [aggressive, moderate,      │
│                  conservative]:             │
│    restore original weights                 │
│    apply strategy (fine-tune model)         │
│    evaluate safety gate                     │
│    if score ≥ 0.7 → PatchResult(passed)     │
│                                             │
│  if all fail:                               │
│    restore original weights                 │
│    PatchResult(passed=False)                │
└─────────────────────────────────────────────┘
        │
        ▼
PatchReport
  └─ PatchResult × num_clusters
```

---

## The 4 patch strategies

Each strategy runs a short fine-tuning loop on the model's last layer (or classifier head for text models). Steps are capped at 8/5/3 for aggressive/moderate/conservative to keep CPU runtime practical.

| Strategy | Mechanism | Best for |
|----------|-----------|----------|
| `adversarial_training` | Generate PGD examples, fine-tune on adversarial + clean mix | Gradient sensitivity (FGSM/PGD clusters) |
| `stylized_augmentation` | Apply channel noise + random grayscale, fine-tune | Texture bias / semantic sensitivity |
| `counterfactual_generation` | Mask border region with random noise, fine-tune | Background dependency / shortcut learning |
| `targeted_augmentation` | Oversample cluster's original inputs + jitter, fine-tune | Edge-case underrepresentation |

All strategies fall back to simple clean fine-tuning for text (DistilBERT) models, where image-specific augmentations don't apply.

---

## Safety gate (D7)

The composite safety score is:

```
safety_score = 0.6 × resistance_gain + 0.4 × accuracy_retention
```

where:
- **resistance_gain**: fraction of the cluster's adversarial examples now correctly classified (0–1). Baseline is effectively 0 since all cluster examples fooled the original model.
- **accuracy_retention**: 1.0 if clean accuracy drop ≤ 3%, otherwise 0.0 (auto-fail).

A patch passes if `safety_score ≥ 0.7`. Since `accuracy_retention` is binary, this requires `resistance_gain ≥ 0.5` (correcting at least half the cluster's adversarial examples) when accuracy holds.

Auto-fail on accuracy drop > 3% catches the robustness-accuracy tradeoff — a patch that breaks clean predictions is not acceptable regardless of how many adversarial examples it fixes.

---

## Retry loop (D8)

Three attempts per cluster, progressively more conservative:

| Attempt | Strength | Steps | LR |
|---------|----------|-------|----|
| 0 (aggressive) | high | 8 | 1e-4 |
| 1 (moderate) | medium | 5 | 5e-5 |
| 2 (conservative) | low | 3 | 1e-5 |

If all three fail the safety gate, the cluster is marked `unresolved`. The model is restored to its pre-patch weights for that cluster. Subsequent clusters are still attempted.

---

## Files

| File | Role |
|------|------|
| `patching/schema.py` | `PatchResult` + `PatchReport` dataclasses |
| `patching/safety_gate.py` | `evaluate()` — composite safety check; `measure_accuracy()` utility |
| `patching/strategies.py` | 4 strategy functions + `STRATEGY_FN` dispatch dict |
| `patching/patcher.py` | `Patcher` class — retry loop orchestrator |
| `patching/__init__.py` | Exports |

---

## Test results

```
tests/test_patching.py — 17/17 passed
```

| Test group | Coverage |
|------------|----------|
| Schema | PatchResult fields, PatchReport summary keys/counts, empty report |
| Safety gate | All-correct accuracy, none-correct accuracy, gate passes on good resistance, gate fails on accuracy drop |
| Strategies | All 4 strategies run without error on 8×8 CPU inputs |
| Patcher | Returns PatchReport, passes on first attempt, retries until pass, marks unresolved after max retries, skips missing cluster |

Strategy functions are tested with real forward/backward passes on a tiny model (3×8×8 inputs, linear classifier). Patcher retry logic is tested with mocked `safety_gate.evaluate` to control outcomes deterministically.

---

## Connection to adjacent phases

- **Receives from Phase 5:** `ExplanationReport` — which strategy to apply per cluster and with what parameters
- **Receives from Phase 4:** `VulnerabilityTaxonomy` — cluster examples needed for resistance measurement
- **Passes to Phase 7:** `PatchReport` — per-cluster patch outcome rendered into the PDF audit report
- **Side effect:** Modifies model weights in place for clusters that pass the safety gate

---

## Key design decisions

### Why pre-defined strategies (see D6 in problems_and_decisions.md)
The space of possible patches is not open-ended. Adversarial ML literature converges on these four as principal remediation approaches. Pre-defining them makes Phase 6 auditable and implementable. A "LLM writes patching code" approach would be unreliable.

### Why composite safety gate (see D7)
Attack resistance alone is insufficient — adversarial training is known to degrade clean accuracy (the robustness-accuracy tradeoff). A patch that makes the model immune to attacks but drops clean accuracy 15% is not acceptable. The composite gate catches this.

### Why 3 retries (see D8)
Three attempts cover the full spectrum of aggression (aggressive → moderate → conservative). A 4th retry would apply such diluted changes it provides no meaningful fix. After 3 failures the vulnerability represents a fundamental tradeoff that cannot be resolved without architectural changes.

### Why fine-tune only the last layer
Fine-tuning all parameters on 3–8 steps with a tiny augmented batch would push the model toward that specific batch, severely degrading clean accuracy. The last layer (fc / classifier) adapts the decision boundary without disturbing the representations learned by earlier layers.
