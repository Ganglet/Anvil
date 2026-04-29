from typing import List
import torch
import torchvision.transforms.functional as TF

from models.base_model import BaseModel
from attacks.base_attack import BaseAttack, AdversarialExample


class SemanticAttack(BaseAttack):
    """
    Semantic perturbations — rotation, brightness, contrast shifts.
    Perturbations are human-perceptible but "natural" (not pixel noise).
    Tests whether the model relies on orientation/lighting rather than semantics.
    """

    # Each tuple: (transform_fn, param_name, values_to_try)
    _TRANSFORMS = [
        ("rotate", [15, 30, -15, -30]),
        ("brightness", [0.5, 1.5, 0.3, 1.8]),
        ("contrast", [0.5, 1.5, 0.3, 2.0]),
    ]

    def __init__(self, model: BaseModel):
        super().__init__(model)

    def _apply(self, x: torch.Tensor, transform: str, value: float) -> torch.Tensor:
        if transform == "rotate":
            return TF.rotate(x, angle=value)
        if transform == "brightness":
            return TF.adjust_brightness(x, brightness_factor=value)
        if transform == "contrast":
            return TF.adjust_contrast(x, contrast_factor=value)
        return x

    def attack(
        self,
        inputs: torch.Tensor,
        labels: List[int],
        profile: dict,
    ) -> List[AdversarialExample]:
        examples = []

        for i, label in enumerate(labels):
            x_orig = inputs[i].unsqueeze(0).detach()

            with torch.no_grad():
                orig_logits = self.model._model(x_orig)
                if hasattr(orig_logits, "logits"):
                    orig_logits = orig_logits.logits
                orig_pred = int(orig_logits.argmax(dim=1).item())

            best_adv = None
            best_pred = orig_pred
            best_params: dict = {}

            # Try every (transform, value) pair; keep first that fools the model
            for transform_name, values in self._TRANSFORMS:
                for value in values:
                    x_t = self._apply(x_orig, transform_name, value)

                    with torch.no_grad():
                        logits = self.model._model(x_t)
                        if hasattr(logits, "logits"):
                            logits = logits.logits
                        pred = int(logits.argmax(dim=1).item())

                    if pred != orig_pred:
                        best_adv = x_t
                        best_pred = pred
                        best_params = {"transform": transform_name, "value": value}
                        break
                if best_adv is not None:
                    break

            # If no transform fooled the model, record the last attempted one
            if best_adv is None:
                x_t = self._apply(x_orig, self._TRANSFORMS[-1][0], self._TRANSFORMS[-1][1][-1])
                best_adv = x_t
                best_params = {
                    "transform": self._TRANSFORMS[-1][0],
                    "value": self._TRANSFORMS[-1][1][-1],
                }

            examples.append(
                AdversarialExample(
                    original_input=inputs[i].detach(),
                    perturbed_input=best_adv.squeeze(0).detach(),
                    true_label=label,
                    original_pred=orig_pred,
                    adversarial_pred=best_pred,
                    attack_name="SemanticAttack",
                    attack_params=best_params,
                )
            )

        return examples
