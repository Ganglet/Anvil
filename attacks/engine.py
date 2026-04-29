from typing import Dict, List
import torch

from models.base_model import BaseModel
from attacks.base_attack import AdversarialExample
from attacks.fgsm import FGSMAttack
from attacks.pgd import PGDAttack
from attacks.patch_attack import PatchAttack
from attacks.semantic_attack import SemanticAttack
from attacks.text_attack import TextAttack


class AttackEngine:
    """
    Orchestrates all attack strategies against a model.
    Selects image vs. text attacks based on model type.
    Returns all AdversarialExamples grouped by attack name.
    """

    def __init__(self, model: BaseModel):
        self.model = model
        self._is_text = "distilbert" in model.model_name

    def run(
        self,
        inputs: torch.Tensor,
        labels: List[int],
        profile: dict,
    ) -> Dict[str, List[AdversarialExample]]:
        """
        Run all applicable attacks and return results keyed by attack name.

        Args:
            inputs:  (N, C, H, W) for images or (N, seq_len) long tensor for text
            labels:  ground-truth class per sample
            profile: Phase 2 profile dict

        Returns:
            {"FGSM": [...], "PGD": [...], ...}
        """
        if self._is_text:
            return self._run_text(inputs, labels, profile)
        return self._run_image(inputs, labels, profile)

    def _run_image(
        self,
        inputs: torch.Tensor,
        labels: List[int],
        profile: dict,
    ) -> Dict[str, List[AdversarialExample]]:
        results: Dict[str, List[AdversarialExample]] = {}

        attacks = [
            FGSMAttack(self.model),
            PGDAttack(self.model),
            PatchAttack(self.model),
            SemanticAttack(self.model),
        ]

        for attack in attacks:
            examples = attack.attack(inputs, labels, profile)
            results[attack.__class__.__name__.replace("Attack", "")] = examples

        return results

    def _run_text(
        self,
        inputs: torch.Tensor,
        labels: List[int],
        profile: dict,
    ) -> Dict[str, List[AdversarialExample]]:
        attack = TextAttack(self.model)
        examples = attack.attack(inputs, labels, profile)
        return {"Text": examples}

    def success_rate(self, results: Dict[str, List[AdversarialExample]]) -> Dict[str, float]:
        """Returns fraction of successful attacks per strategy."""
        rates = {}
        for name, examples in results.items():
            if examples:
                rates[name] = sum(1 for e in examples if e.success) / len(examples)
            else:
                rates[name] = 0.0
        return rates
