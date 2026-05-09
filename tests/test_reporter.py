import os
import tempfile
import numpy as np
import pytest
import torch

from attacks.base_attack import AdversarialExample
from clustering.taxonomy import VulnerabilityCluster, VulnerabilityTaxonomy
from agent.schema import ClusterExplanation, ExplanationReport
from patching.schema import PatchResult, PatchReport
from reporter.report import generate_report


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _profile():
    return {
        "model": "resnet18",
        "num_samples": 10,
        "vulnerability_score": 0.42,
        "mean_saliency_score": 0.31,
        "attack_priority": ["layer4", "layer1", "avgpool"],
        "gradient_norms": {"layer4": 0.85, "layer1": 0.62, "avgpool": 0.34},
        "activation_entropy": {"layer4": 1.2, "layer1": 0.9, "avgpool": 0.5},
    }


def _attack_rates():
    return {"FGSM": 1.0, "PGD": 1.0, "Patch": 0.1, "Semantic": 1.0}


def _example():
    return AdversarialExample(
        original_input=torch.rand(3, 224, 224),
        perturbed_input=torch.rand(3, 224, 224),
        true_label=0, original_pred=0, adversarial_pred=1, attack_name="FGSM",
    )


def _taxonomy(num_clusters=2):
    clusters = []
    for i in range(num_clusters):
        clusters.append(VulnerabilityCluster(
            cluster_id=i,
            name=f"FGSM_vulnerability_{i}",
            dominant_attack="FGSM",
            size=20,
            centroid=np.array([0.1, 0.2]),
            examples=[_example()],
            attack_distribution={"FGSM": 20},
        ))
    return VulnerabilityTaxonomy(
        clusters=clusters, noise_count=2,
        total_failures=40 + 2, model_name="resnet18",
    )


def _explanation_report(num_clusters=2):
    explanations = []
    for i in range(num_clusters):
        explanations.append(ClusterExplanation(
            cluster_id=i,
            cluster_name=f"FGSM_vulnerability_{i}",
            explanation=(
                "This cluster represents gradient-based sensitivity. "
                "The model responds strongly to small perturbations in the gradient direction, "
                "consistent with findings from Goodfellow et al. (2014). "
                "Adversarial training is the recommended remediation."
            ),
            patch_strategy="adversarial_training",
            patch_params={"layers": "all", "strength": "high", "steps": 100},
            sources=["Goodfellow 2014 — FGSM", "Madry 2018 — PGD"],
        ))
    return ExplanationReport(model_name="resnet18", explanations=explanations)


def _patch_report(passed=False, num_clusters=2):
    results = []
    for i in range(num_clusters):
        results.append(PatchResult(
            cluster_id=i,
            cluster_name=f"FGSM_vulnerability_{i}",
            strategy="adversarial_training",
            safety_score=0.82 if passed else 0.0,
            passed=passed,
            retries=0 if passed else 3,
            resistance_gain=0.7 if passed else 0.1,
            accuracy_drop=0.01 if passed else 0.08,
        ))
    return PatchReport(model_name="resnet18", results=results)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_report_creates_file():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "report.pdf")
        generate_report(
            output_path=out,
            model_name="resnet18",
            profile=_profile(),
            attack_rates=_attack_rates(),
            total_fooled=62,
            total_examples=80,
            taxonomy=_taxonomy(),
            explanation_report=_explanation_report(),
            patch_report=_patch_report(),
        )
        assert os.path.exists(out)


def test_report_file_not_empty():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "report.pdf")
        generate_report(
            output_path=out,
            model_name="resnet18",
            profile=_profile(),
            attack_rates=_attack_rates(),
            total_fooled=62,
            total_examples=80,
            taxonomy=_taxonomy(),
            explanation_report=_explanation_report(),
            patch_report=_patch_report(),
        )
        assert os.path.getsize(out) > 5_000


def test_report_uses_output_path():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "my_audit.pdf")
        generate_report(
            output_path=out,
            model_name="resnet18",
            profile=_profile(),
            attack_rates=_attack_rates(),
            total_fooled=10,
            total_examples=20,
            taxonomy=_taxonomy(),
            explanation_report=_explanation_report(),
            patch_report=_patch_report(),
        )
        assert os.path.exists(out)
        assert not os.path.exists(os.path.join(tmp, "report.pdf"))


def test_report_no_clusters():
    empty_taxonomy = VulnerabilityTaxonomy(
        clusters=[], noise_count=5, total_failures=5, model_name="resnet18",
    )
    empty_exp = ExplanationReport(model_name="resnet18", explanations=[])
    empty_patch = PatchReport(model_name="resnet18", results=[])

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "report.pdf")
        generate_report(
            output_path=out,
            model_name="resnet18",
            profile=_profile(),
            attack_rates=_attack_rates(),
            total_fooled=5,
            total_examples=20,
            taxonomy=empty_taxonomy,
            explanation_report=empty_exp,
            patch_report=empty_patch,
        )
        assert os.path.exists(out)
        assert os.path.getsize(out) > 5_000


def test_report_with_passed_patches():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "report.pdf")
        generate_report(
            output_path=out,
            model_name="resnet18",
            profile=_profile(),
            attack_rates=_attack_rates(),
            total_fooled=62,
            total_examples=80,
            taxonomy=_taxonomy(),
            explanation_report=_explanation_report(),
            patch_report=_patch_report(passed=True),
        )
        assert os.path.getsize(out) > 5_000


def test_report_single_cluster():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "report.pdf")
        generate_report(
            output_path=out,
            model_name="resnet18",
            profile=_profile(),
            attack_rates={"FGSM": 0.9},
            total_fooled=18,
            total_examples=20,
            taxonomy=_taxonomy(num_clusters=1),
            explanation_report=_explanation_report(num_clusters=1),
            patch_report=_patch_report(num_clusters=1),
        )
        assert os.path.exists(out)


def test_report_low_vulnerability_score():
    profile = _profile()
    profile["vulnerability_score"] = 0.1
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "report.pdf")
        generate_report(
            output_path=out,
            model_name="distilbert",
            profile=profile,
            attack_rates={"Text": 0.2},
            total_fooled=4,
            total_examples=20,
            taxonomy=_taxonomy(num_clusters=1),
            explanation_report=_explanation_report(num_clusters=1),
            patch_report=_patch_report(num_clusters=1),
        )
        assert os.path.getsize(out) > 5_000
