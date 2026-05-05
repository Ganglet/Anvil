# Phase 4 — Failure Mode Clustering

## What this phase does

Phase 3 produced a flat list of `AdversarialExample` objects — hundreds of perturbed inputs where the model was (or wasn't) fooled. Phase 4 asks a different question: **do the successful failures have patterns?**

A model might fail because of texture sensitivity, brightness dependence, patch placement, or edge-case underrepresentation. If FGSM and PGD failures all cluster together with similar internal activations, that signals a gradient-exploitation weakness — not random noise sensitivity. Phase 4 finds and names these patterns as a `VulnerabilityTaxonomy`.

The taxonomy is the input to Phase 5's LangGraph LLM agent, which writes a human-readable explanation for each cluster and maps it to a remediation strategy.

---

## Pipeline

```
AdversarialExample list (Phase 3)
        │
        ▼
FeatureExtractor
  ├─ filter: keep only successful attacks (success=True)
  ├─ run perturbed_input through top-ranked layer (from Phase 2 attack_priority)
  └─ pool activations → 1D vector per example
        │
        ▼
(N, D) feature matrix
        │
        ▼
FailureModeClusterer
  ├─ UMAP: (N, D) → (N, 5)   — compress while preserving manifold structure
  └─ HDBSCAN: (N, 5) → cluster labels   — density-based, no fixed k
        │
        ▼
VulnerabilityTaxonomy
  ├─ VulnerabilityCluster × num_clusters
  └─ noise_count (failures that fit no cluster)
```

---

## Key design decisions

### Why activation vectors, not raw pixel differences

The raw pixel difference between original and perturbed image tells you *how* the attack perturbed the input. The activation vector from an intermediate layer tells you *how the model internally processed* that perturbed input — which neurons fired, which features were activated. Two very different-looking perturbations can produce nearly identical activations if they exploit the same internal vulnerability. Clustering in activation space groups failures by the same root cause.

The specific layer is chosen by Phase 2's `attack_priority` list — the highest-ranked layer (highest gradient norm + lowest activation entropy) is the most vulnerable layer, so its activations carry the most signal about why the model failed.

### Why UMAP + HDBSCAN (see D1 in problems_and_decisions.md)

- **UMAP over PCA:** Neural network activations lie on a non-linear manifold. PCA's linear projection collapses curved neighbourhood structure. UMAP preserves it.
- **HDBSCAN over k-means:** We don't know the number of vulnerability types in advance — that's what we're discovering. HDBSCAN finds the number automatically and correctly designates anomalous failures as noise rather than forcing them into the nearest cluster.

### Activation pooling

Raw activation tensors have spatial or sequence dimensions:
- ResNet layer4: `(1, 512, 7, 7)` → global average pool → `(512,)`
- DistilBERT transformer layer: `(1, seq_len, 768)` → mean over sequence → `(768,)`

Pooling collapses these to a 1D vector per example, giving UMAP a consistent input shape regardless of model type.

---

## Files

| File | Role |
|------|------|
| `clustering/taxonomy.py` | `VulnerabilityCluster` + `VulnerabilityTaxonomy` dataclasses; `.summary()` produces the dict Phase 5 reads |
| `clustering/feature_extractor.py` | Extracts and pools activation vectors from successful adversarial examples |
| `clustering/clusterer.py` | UMAP reduction → HDBSCAN labeling → named `VulnerabilityCluster` objects |
| `clustering/__init__.py` | Exports all three classes |

---

## Output format

`VulnerabilityTaxonomy.summary()` returns:

```json
{
  "model": "resnet18",
  "total_failures": 47,
  "num_clusters": 3,
  "noise_count": 5,
  "clusters": [
    {
      "id": 0,
      "name": "FGSM_vulnerability_0",
      "size": 21,
      "dominant_attack": "FGSM",
      "attack_distribution": {"FGSM": 18, "PGD": 3}
    },
    ...
  ]
}
```

Cluster names use the format `{dominant_attack}_vulnerability_{id}`. Phase 5's LLM agent replaces these with semantically meaningful names (e.g. `"gradient_sensitivity_texture_artifacts"`) based on the centroid's position in activation space and retrieved paper context.

---

## Test results

```
tests/test_clustering.py — 16/16 passed
```

| Test group | Coverage |
|------------|----------|
| FeatureExtractor | Shape, dtype, filtering (success-only), empty input, missing priority |
| FailureModeClusterer | Taxonomy structure, cluster naming, dominant attack, centroid shape, edge cases (0 and 1 example) |
| VulnerabilityTaxonomy | Summary keys, cluster fields, empty cluster case |

---

## Connection to adjacent phases

- **Receives from Phase 3:** `List[AdversarialExample]` — the complete set of attack attempts across all strategies
- **Passes to Phase 5:** `VulnerabilityTaxonomy` — cluster names, sizes, centroids, and attack distributions that drive the LLM explanation and remediation mapping
