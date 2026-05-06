import os
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from agent.schema import ClusterExplanation, ExplanationReport
from agent.retriever import PaperRetriever
from agent.nodes import _parse_recommendation, retrieve_node, explain_node, recommend_node
from clustering.taxonomy import VulnerabilityCluster, VulnerabilityTaxonomy
from attacks.base_attack import AdversarialExample
import torch


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_cluster(cluster_id: int = 0, dominant: str = "FGSM") -> VulnerabilityCluster:
    ex = AdversarialExample(
        original_input=torch.rand(3, 224, 224),
        perturbed_input=torch.rand(3, 224, 224),
        true_label=0,
        original_pred=0,
        adversarial_pred=1,
        attack_name=dominant,
    )
    return VulnerabilityCluster(
        cluster_id=cluster_id,
        name=f"{dominant}_vulnerability_{cluster_id}",
        dominant_attack=dominant,
        size=10,
        centroid=np.array([0.1, 0.2]),
        examples=[ex],
        attack_distribution={dominant: 10},
    )


def _make_taxonomy() -> VulnerabilityTaxonomy:
    return VulnerabilityTaxonomy(
        clusters=[_make_cluster(0, "FGSM"), _make_cluster(1, "Semantic")],
        noise_count=2,
        total_failures=22,
        model_name="resnet18",
    )


# ── Schema ────────────────────────────────────────────────────────────────────

def test_cluster_explanation_fields():
    exp = ClusterExplanation(
        cluster_id=0,
        cluster_name="FGSM_vulnerability_0",
        explanation="The model is vulnerable to gradient-based attacks.",
        patch_strategy="adversarial_training",
        patch_params={"layers": "all", "strength": "medium", "steps": 100},
        sources=["Goodfellow 2014 — FGSM"],
    )
    assert exp.cluster_id == 0
    assert exp.patch_strategy == "adversarial_training"
    assert len(exp.sources) == 1


def test_explanation_report_summary_keys():
    exp = ClusterExplanation(
        cluster_id=0,
        cluster_name="FGSM_vulnerability_0",
        explanation="test",
        patch_strategy="adversarial_training",
        patch_params={},
        sources=["Paper A"],
    )
    report = ExplanationReport(model_name="resnet18", explanations=[exp])
    s = report.summary()
    assert set(s.keys()) == {"model", "num_clusters_explained", "clusters"}
    assert s["num_clusters_explained"] == 1
    assert s["clusters"][0]["patch_strategy"] == "adversarial_training"


def test_explanation_report_empty():
    report = ExplanationReport(model_name="resnet18", explanations=[])
    s = report.summary()
    assert s["num_clusters_explained"] == 0
    assert s["clusters"] == []


# ── _parse_recommendation ─────────────────────────────────────────────────────

def test_parse_recommendation_valid():
    text = (
        "STRATEGY: adversarial_training\n"
        "PARAM_LAYERS: layer3,layer4\n"
        "PARAM_STRENGTH: high\n"
        "PARAM_STEPS: 200"
    )
    result = _parse_recommendation(text)
    assert result["strategy"] == "adversarial_training"
    assert result["layers"] == "layer3,layer4"
    assert result["strength"] == "high"
    assert result["steps"] == 200


def test_parse_recommendation_invalid_strategy_falls_back():
    text = "STRATEGY: magic_fix\nPARAM_STRENGTH: medium\nPARAM_STEPS: 50"
    result = _parse_recommendation(text)
    assert result["strategy"] == "adversarial_training"


def test_parse_recommendation_invalid_steps_falls_back():
    text = "STRATEGY: targeted_augmentation\nPARAM_STEPS: not_a_number"
    result = _parse_recommendation(text)
    assert result["steps"] == 100


def test_parse_recommendation_all_strategies():
    for strategy in ["adversarial_training", "targeted_augmentation",
                     "counterfactual_generation", "stylized_augmentation"]:
        text = f"STRATEGY: {strategy}\nPARAM_STRENGTH: low\nPARAM_STEPS: 50"
        result = _parse_recommendation(text)
        assert result["strategy"] == strategy


# ── PaperRetriever ────────────────────────────────────────────────────────────

def test_retriever_returns_top_k():
    retriever = PaperRetriever(top_k=3)
    results = retriever.retrieve("gradient sensitivity FGSM adversarial attack")
    assert len(results) == 3


def test_retriever_result_keys():
    retriever = PaperRetriever(top_k=1)
    results = retriever.retrieve("texture bias shortcut learning")
    assert set(results[0].keys()) == {"source", "text", "score"}


def test_retriever_score_in_range():
    retriever = PaperRetriever(top_k=5)
    results = retriever.retrieve("adversarial patch universal attack")
    for r in results:
        assert 0.0 <= r["score"] <= 1.0


def test_retriever_sources_are_named_papers():
    retriever = PaperRetriever(top_k=3)
    results = retriever.retrieve("PGD adversarial training robustness")
    for r in results:
        assert len(r["source"]) > 0


# ── LangGraph nodes (mocked LLM) ──────────────────────────────────────────────

def _mock_llm(response_text: str) -> MagicMock:
    llm = MagicMock()
    msg = MagicMock()
    msg.content = response_text
    llm.invoke.return_value = msg
    return llm


def test_retrieve_node_populates_chunks():
    retriever = PaperRetriever(top_k=3)
    cluster = _make_cluster()
    state = {"retriever": retriever, "cluster": cluster, "chunks": []}
    out = retrieve_node(state)
    assert len(out["chunks"]) == 3
    assert all("text" in c for c in out["chunks"])


def test_explain_node_sets_explanation():
    cluster = _make_cluster()
    chunks = [{"source": "Goodfellow 2014", "text": "FGSM adds gradient sign perturbation.", "score": 0.9}]
    state = {
        "llm": _mock_llm("The model is sensitive to gradient-based perturbations."),
        "cluster": cluster,
        "chunks": chunks,
        "explanation": "",
    }
    out = explain_node(state)
    assert len(out["explanation"]) > 0


def test_recommend_node_returns_cluster_explanation():
    cluster = _make_cluster()
    rec_text = (
        "STRATEGY: adversarial_training\n"
        "PARAM_LAYERS: layer4\n"
        "PARAM_STRENGTH: high\n"
        "PARAM_STEPS: 150"
    )
    state = {
        "llm": _mock_llm(rec_text),
        "cluster": cluster,
        "chunks": [{"source": "Madry 2018", "text": "PGD training.", "score": 0.8}],
        "explanation": "Vulnerable to gradient attacks.",
        "result": None,
    }
    out = recommend_node(state)
    result = out["result"]
    assert isinstance(result, ClusterExplanation)
    assert result.patch_strategy == "adversarial_training"
    assert result.patch_params["steps"] == 150
    assert "Madry 2018" in result.sources


def test_recommend_node_deduplicates_sources():
    cluster = _make_cluster()
    rec_text = "STRATEGY: targeted_augmentation\nPARAM_LAYERS: all\nPARAM_STRENGTH: medium\nPARAM_STEPS: 100"
    chunks = [
        {"source": "Goodfellow 2014", "text": "a", "score": 0.9},
        {"source": "Goodfellow 2014", "text": "b", "score": 0.8},
        {"source": "Madry 2018", "text": "c", "score": 0.7},
    ]
    state = {
        "llm": _mock_llm(rec_text),
        "cluster": cluster,
        "chunks": chunks,
        "explanation": "test",
        "result": None,
    }
    out = recommend_node(state)
    assert len(out["result"].sources) == 2
