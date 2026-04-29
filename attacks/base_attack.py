from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List
import torch

from models.base_model import BaseModel


@dataclass
class AdversarialExample:
    original_input: torch.Tensor
    perturbed_input: torch.Tensor
    true_label: int
    original_pred: int
    adversarial_pred: int
    attack_name: str
    attack_params: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.original_pred != self.adversarial_pred


class BaseAttack(ABC):
    def __init__(self, model: BaseModel):
        self.model = model

    @abstractmethod
    def attack(
        self,
        inputs: torch.Tensor,
        labels: List[int],
        profile: dict,
    ) -> List[AdversarialExample]:
        """
        Run the attack on a batch of inputs.

        Args:
            inputs:  (N, C, H, W) for images or (N, seq_len) for text
            labels:  ground-truth class per sample
            profile: Phase 2 profile dict (attack_priority, gradient_norms, etc.)

        Returns:
            List of AdversarialExample — one per input sample.
        """
