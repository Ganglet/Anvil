import copy
from unittest.mock import MagicMock, patch
from typing import List
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel
from attacks.base_attack import AdversarialExample
from clustering.taxonomy import VulnerabilityCluster, VulnerabilityTaxonomy
from agent.schema import ClusterExplanation, ExplanationReport
from patching.schema import PatchResult, PatchReport
from patching import safety_gate
from patching.strategies import (
    adversarial_training, stylized_augmentation,
    counterfactual_generation, targeted_augmentation,
)
from patching.patcher import Patcher


# ── Minimal model for fast CPU tests ─────────────────────────────────────────

class _TinyModel(BaseModel):
    """8×8 input, 10-class linear model — fast on CPU."""

    def __init__(self, always_predict: int = None):
        self._always_predict = always_predict
        self._model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 10))
        self._model.eval()

    @property
    def model_name(self) -> str:
        return "tiny"

    @property
    def input_shape(self) -> tuple:
        return (3, 8, 8)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        if self._always_predict is not None:
            n = inputs.shape[0]
            logits = torch.zeros(n, 10)
            logits[:, self._always_predict] = 10.0
            return logits
        with torch.no_grad():
            return self._model(inputs.float())

    def get_gradients(self, inputs, target_class):
        return torch.zeros_like(inputs)

    def get_activations(self, inputs, layer_name):
        return self._model(inputs.float())


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_example(true_label: int = 0, fooled: bool = True) -> AdversarialExample:
    return AdversarialExample(
        original_input=torch.rand(3, 8, 8),
        perturbed_input=torch.rand(3, 8, 8),
        true_label=true_label,
        original_pred=true_label,
        adversarial_pred=true_label + 1 if fooled else true_label,
        attack_name="FGSM",
    )


def _make_cluster(cluster_id: int = 0, size: int = 3) -> VulnerabilityCluster:
    examples = [_make_example(true_label=0, fooled=True) for _ in range(size)]
    return VulnerabilityCluster(
        cluster_id=cluster_id,
        name=f"FGSM_vulnerability_{cluster_id}",
        dominant_attack="FGSM",
        size=size,
        centroid=np.array([0.1, 0.2]),
        examples=examples,
        attack_distribution={"FGSM": size},
    )


def _make_explanation(
    cluster_id: int = 0, strategy: str = "adversarial_training"
) -> ClusterExplanation:
    return ClusterExplanation(
        cluster_id=cluster_id,
        cluster_name=f"FGSM_vulnerability_{cluster_id}",
        explanation="Vulnerable to gradient-based attacks.",
        patch_strategy=strategy,
        patch_params={"layers": "all", "strength": "high", "steps": 8},
        sources=["Goodfellow 2014"],
    )


def _clean_batch(n: int = 4):
    inputs = torch.rand(n, 3, 8, 8)
    labels = [0] * n
    return inputs, labels


# ── Schema tests ──────────────────────────────────────────────────────────────

def test_patch_result_fields():
    r = PatchResult(
        cluster_id=0, cluster_name="x", strategy="adversarial_training",
        safety_score=0.82, passed=True, retries=0,
        resistance_gain=0.7, accuracy_drop=0.01,
    )
    assert r.passed is True
    assert r.retries == 0
    assert r.safety_score == 0.82


def test_patch_report_summary_keys():
    r = PatchResult(0, "x", "adversarial_training", 0.82, True, 0, 0.7, 0.01)
    report = PatchReport(model_name="resnet18", results=[r])
    s = report.summary()
    assert set(s.keys()) == {"model", "total_clusters", "patched", "unresolved", "clusters"}


def test_patch_report_summary_counts():
    passed = PatchResult(0, "a", "adversarial_training", 0.85, True, 0, 0.8, 0.01)
    failed = PatchResult(1, "b", "stylized_augmentation", 0.3, False, 3, 0.1, 0.05)
    report = PatchReport(model_name="resnet18", results=[passed, failed])
    s = report.summary()
    assert s["patched"] == 1
    assert s["unresolved"] == 1
    assert s["total_clusters"] == 2


def test_patch_report_empty():
    report = PatchReport(model_name="resnet18", results=[])
    s = report.summary()
    assert s["patched"] == 0
    assert s["total_clusters"] == 0


# ── Safety gate tests ─────────────────────────────────────────────────────────

def test_measure_accuracy_all_correct():
    model = _TinyModel(always_predict=0)
    inputs, labels = _clean_batch(4)
    acc = safety_gate.measure_accuracy(model, inputs, labels)
    assert acc == 1.0


def test_measure_accuracy_none_correct():
    model = _TinyModel(always_predict=1)
    inputs = torch.rand(4, 3, 8, 8)
    labels = [0, 0, 0, 0]
    acc = safety_gate.measure_accuracy(model, inputs, labels)
    assert acc == 0.0


def test_safety_gate_passes_with_good_resistance():
    # Model predicts true_label=0 for all examples → resistance_gain = 1.0
    # safety_score = 0.6 * 1.0 + 0.4 * 1.0 = 1.0
    model = _TinyModel(always_predict=0)
    cluster = _make_cluster()  # all examples have true_label=0
    inputs, labels = _clean_batch()
    baseline = 1.0  # model already correct on clean inputs
    score, resistance, drop = safety_gate.evaluate(model, cluster, inputs, labels, baseline)
    assert score >= safety_gate.PASS_THRESHOLD
    assert resistance == 1.0
    assert drop == 0.0


def test_safety_gate_fails_on_accuracy_drop():
    # Baseline = 1.0, post = 0.0 → drop = 1.0 > 0.03 → auto-fail
    model_before = _TinyModel(always_predict=0)
    model_after = _TinyModel(always_predict=1)
    cluster = _make_cluster()
    inputs = torch.rand(4, 3, 8, 8)
    labels = [0, 0, 0, 0]
    baseline = safety_gate.measure_accuracy(model_before, inputs, labels)  # 1.0
    score, _, drop = safety_gate.evaluate(model_after, cluster, inputs, labels, baseline)
    assert score == 0.0
    assert drop > safety_gate.MAX_ACCURACY_DROP


# ── Strategy tests (just run without errors) ──────────────────────────────────

def test_adversarial_training_runs():
    model = _TinyModel()
    cluster = _make_cluster()
    inputs, labels = _clean_batch()
    adversarial_training(model, cluster, inputs, labels, "high", 3)
    assert model._model is not None


def test_stylized_augmentation_runs():
    model = _TinyModel()
    cluster = _make_cluster()
    inputs, labels = _clean_batch()
    stylized_augmentation(model, cluster, inputs, labels, "medium", 3)
    assert model._model is not None


def test_counterfactual_generation_runs():
    model = _TinyModel()
    cluster = _make_cluster()
    inputs, labels = _clean_batch()
    counterfactual_generation(model, cluster, inputs, labels, "low", 3)
    assert model._model is not None


def test_targeted_augmentation_runs():
    model = _TinyModel()
    cluster = _make_cluster()
    inputs, labels = _clean_batch()
    targeted_augmentation(model, cluster, inputs, labels, "medium", 3)
    assert model._model is not None


# ── Patcher integration tests ─────────────────────────────────────────────────

def _make_taxonomy_and_report(strategy: str = "adversarial_training"):
    cluster = _make_cluster(0)
    taxonomy = VulnerabilityTaxonomy(
        clusters=[cluster], noise_count=0,
        total_failures=3, model_name="tiny",
    )
    explanation = _make_explanation(0, strategy)
    report = ExplanationReport(model_name="tiny", explanations=[explanation])
    return taxonomy, report


def test_patcher_returns_patch_report():
    model = _TinyModel()
    taxonomy, report = _make_taxonomy_and_report()
    inputs, labels = _clean_batch()

    with patch("patching.patcher.safety_gate.evaluate", return_value=(0.85, 0.75, 0.01)):
        patch_report = Patcher().patch(model, taxonomy, report, inputs, labels)

    assert isinstance(patch_report, PatchReport)
    assert len(patch_report.results) == 1


def test_patcher_passes_on_first_attempt():
    model = _TinyModel()
    taxonomy, report = _make_taxonomy_and_report()
    inputs, labels = _clean_batch()

    with patch("patching.patcher.safety_gate.evaluate", return_value=(0.85, 0.75, 0.01)):
        with patch("patching.patcher.STRATEGY_FN", {"adversarial_training": MagicMock()}):
            patch_report = Patcher().patch(model, taxonomy, report, inputs, labels)

    result = patch_report.results[0]
    assert result.passed is True
    assert result.retries == 0


def test_patcher_retries_until_pass():
    model = _TinyModel()
    taxonomy, report = _make_taxonomy_and_report()
    inputs, labels = _clean_batch()

    # Fail first two attempts, pass on third
    side_effects = [(0.3, 0.1, 0.01), (0.5, 0.3, 0.01), (0.8, 0.7, 0.01)]
    with patch("patching.patcher.safety_gate.evaluate", side_effect=side_effects):
        with patch("patching.patcher.STRATEGY_FN", {"adversarial_training": MagicMock()}):
            patch_report = Patcher().patch(model, taxonomy, report, inputs, labels)

    result = patch_report.results[0]
    assert result.passed is True
    assert result.retries == 2


def test_patcher_marks_unresolved_after_max_retries():
    model = _TinyModel()
    taxonomy, report = _make_taxonomy_and_report()
    inputs, labels = _clean_batch()

    with patch("patching.patcher.safety_gate.evaluate", return_value=(0.3, 0.1, 0.01)):
        with patch("patching.patcher.STRATEGY_FN", {"adversarial_training": MagicMock()}):
            patch_report = Patcher().patch(model, taxonomy, report, inputs, labels)

    result = patch_report.results[0]
    assert result.passed is False
    assert result.retries == 3


def test_patcher_skips_missing_cluster():
    # Explanation references cluster_id=99 which isn't in taxonomy
    cluster = _make_cluster(0)
    taxonomy = VulnerabilityTaxonomy(
        clusters=[cluster], noise_count=0, total_failures=3, model_name="tiny",
    )
    explanation = _make_explanation(cluster_id=99)
    report = ExplanationReport(model_name="tiny", explanations=[explanation])
    inputs, labels = _clean_batch()

    patch_report = Patcher().patch(_TinyModel(), taxonomy, report, inputs, labels)
    assert len(patch_report.results) == 0
