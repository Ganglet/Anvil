from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torchvision import models, transforms

from .base_model import BaseModel


class ImageModel(BaseModel):
    """
    ResNet-18 wrapper. Pretrained on ImageNet (1000 classes).
    Supports gradient extraction for FGSM/PGD and activation hooks for clustering.
    """

    def __init__(self, pretrained: bool = True, num_classes: int = 1000):
        self._model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        if num_classes != 1000:
            self._model.fc = nn.Linear(self._model.fc.in_features, num_classes)

        self._model.eval()
        self._activation_cache: Dict[str, torch.Tensor] = {}
        self._hooks: List = []

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    @property
    def model_name(self) -> str:
        return "resnet18"

    @property
    def input_shape(self) -> tuple:
        return (3, 224, 224)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        self._model.eval()
        with torch.no_grad():
            return self._model(inputs)

    def get_gradients(
        self, inputs: torch.Tensor, target_class: int
    ) -> torch.Tensor:
        self._model.eval()
        x = inputs.clone().detach().requires_grad_(True)
        logits = self._model(x)
        self._model.zero_grad()
        logits[0, target_class].backward()
        return x.grad.detach()

    def get_activations(
        self, inputs: torch.Tensor, layer_name: str
    ) -> torch.Tensor:
        self._clear_hooks()
        self._activation_cache = {}

        target_layer = dict(self._model.named_modules()).get(layer_name)
        if target_layer is None:
            raise ValueError(
                f"Layer '{layer_name}' not found. "
                f"Available: {self.get_available_layers()}"
            )

        def hook_fn(module, input, output):
            self._activation_cache[layer_name] = output.detach()

        handle = target_layer.register_forward_hook(hook_fn)
        self._hooks.append(handle)

        self._model.eval()
        with torch.no_grad():
            self._model(inputs)

        self._clear_hooks()
        return self._activation_cache[layer_name]

    def get_available_layers(self) -> List[str]:
        return [name for name, _ in self._model.named_modules() if name]

    def _clear_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
