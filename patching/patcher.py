import copy
from typing import List
import torch

from models.base_model import BaseModel
from clustering.taxonomy import VulnerabilityTaxonomy, VulnerabilityCluster
from agent.schema import ExplanationReport, ClusterExplanation
from patching.schema import PatchResult, PatchReport
from patching import safety_gate
from patching.strategies import STRATEGY_FN

_RETRY_CONFIGS = [
    {"strength": "high",   "steps": 8},
    {"strength": "medium", "steps": 5},
    {"strength": "low",    "steps": 3},
]


class Patcher:
    def patch(
        self,
        model: BaseModel,
        taxonomy: VulnerabilityTaxonomy,
        explanation_report: ExplanationReport,
        clean_inputs: torch.Tensor,
        clean_labels: List[int],
    ) -> PatchReport:
        cluster_map = {c.cluster_id: c for c in taxonomy.clusters}
        results = []

        for explanation in explanation_report.explanations:
            cluster = cluster_map.get(explanation.cluster_id)
            if cluster is None:
                continue
            result = self._patch_cluster(
                model, cluster, explanation, clean_inputs, clean_labels
            )
            results.append(result)

        return PatchReport(model_name=model.model_name, results=results)

    def _patch_cluster(
        self,
        model: BaseModel,
        cluster: VulnerabilityCluster,
        explanation: ClusterExplanation,
        clean_inputs: torch.Tensor,
        clean_labels: List[int],
    ) -> PatchResult:
        original_state = copy.deepcopy(model._model.state_dict())
        baseline_acc = safety_gate.measure_accuracy(model, clean_inputs, clean_labels)

        strategy_fn = STRATEGY_FN.get(
            explanation.patch_strategy, STRATEGY_FN["adversarial_training"]
        )

        last_score = 0.0
        last_resistance = 0.0
        last_drop = 0.0

        for attempt, config in enumerate(_RETRY_CONFIGS):
            model._model.load_state_dict(original_state)

            try:
                strategy_fn(
                    model,
                    cluster,
                    clean_inputs,
                    clean_labels,
                    config["strength"],
                    config["steps"],
                )
            except Exception:
                continue

            score, resistance, drop = safety_gate.evaluate(
                model, cluster, clean_inputs, clean_labels, baseline_acc
            )
            last_score = score
            last_resistance = resistance
            last_drop = drop

            if score >= safety_gate.PASS_THRESHOLD:
                return PatchResult(
                    cluster_id=cluster.cluster_id,
                    cluster_name=cluster.name,
                    strategy=explanation.patch_strategy,
                    safety_score=round(score, 4),
                    passed=True,
                    retries=attempt,
                    resistance_gain=round(resistance, 4),
                    accuracy_drop=round(drop, 4),
                )

        # All retries failed — restore original weights
        model._model.load_state_dict(original_state)
        return PatchResult(
            cluster_id=cluster.cluster_id,
            cluster_name=cluster.name,
            strategy=explanation.patch_strategy,
            safety_score=round(last_score, 4),
            passed=False,
            retries=len(_RETRY_CONFIGS),
            resistance_gain=round(last_resistance, 4),
            accuracy_drop=round(last_drop, 4),
        )
