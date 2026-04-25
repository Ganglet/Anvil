from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import torch


class BaseModel(ABC):
    """
    Model-agnostic interface. Every model Red Queen attacks must implement these
    three methods — predict, get_gradients, get_activations. Phases 3-6 depend on them.
    """

    @abstractmethod
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Returns raw logits (not softmax).
        Shape: (batch_size, num_classes)
        """
        pass

    @abstractmethod
    def get_gradients(
        self, inputs: torch.Tensor, target_class: int
    ) -> torch.Tensor:
        """
        Gradient of the target class logit w.r.t. the input tensor.
        Used by FGSM and PGD in Phase 3.
        Returns tensor of same shape as inputs.
        """
        pass

    @abstractmethod
    def get_activations(
        self, inputs: torch.Tensor, layer_name: str
    ) -> torch.Tensor:
        """
        Intermediate layer activations for the given layer_name.
        Used by UMAP+HDBSCAN clustering in Phase 4.
        Returns tensor of shape (batch_size, *layer_output_dims).
        """
        pass

    def get_available_layers(self) -> List[str]:
        """Returns layer names that can be passed to get_activations."""
        return []

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def input_shape(self) -> tuple:
        """Expected input shape (C, H, W) for images or (seq_len,) for text."""
        pass
