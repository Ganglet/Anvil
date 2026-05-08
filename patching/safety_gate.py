from typing import List, Tuple
import torch

from models.base_model import BaseModel
from clustering.taxonomy import VulnerabilityCluster

RESISTANCE_WEIGHT = 0.6
ACCURACY_WEIGHT = 0.4
PASS_THRESHOLD = 0.7
MAX_ACCURACY_DROP = 0.03


def evaluate(
    model: BaseModel,
    cluster: VulnerabilityCluster,
    clean_inputs: torch.Tensor,
    clean_labels: List[int],
    baseline_accuracy: float,
) -> Tuple[float, float, float]:
    """
    Returns (safety_score, resistance_gain, accuracy_drop).
    safety_score >= PASS_THRESHOLD means the patch passes.
    Auto-fails (score=0) if clean accuracy drops more than 3%.
    """
    resistance_gain = _measure_resistance(model, cluster)
    post_accuracy = measure_accuracy(model, clean_inputs, clean_labels)
    accuracy_drop = baseline_accuracy - post_accuracy

    if accuracy_drop > MAX_ACCURACY_DROP:
        return 0.0, resistance_gain, accuracy_drop

    safety_score = RESISTANCE_WEIGHT * resistance_gain + ACCURACY_WEIGHT * 1.0
    return safety_score, resistance_gain, accuracy_drop


def measure_accuracy(
    model: BaseModel, inputs: torch.Tensor, labels: List[int]
) -> float:
    """Fraction of clean inputs correctly classified."""
    if len(inputs) == 0 or len(labels) == 0:
        return 1.0
    with torch.no_grad():
        logits = model.predict(inputs)
    preds = logits.argmax(dim=1).tolist()
    correct = sum(p == l for p, l in zip(preds, labels))
    return correct / len(labels)


def _measure_resistance(model: BaseModel, cluster: VulnerabilityCluster) -> float:
    """Fraction of cluster's adversarial examples now correctly classified."""
    if not cluster.examples:
        return 0.0
    correct = 0
    for ex in cluster.examples:
        inp = ex.perturbed_input.unsqueeze(0)
        with torch.no_grad():
            logits = model.predict(inp)
        pred = int(logits.argmax(dim=1).item())
        if pred == ex.true_label:
            correct += 1
    return correct / len(cluster.examples)
