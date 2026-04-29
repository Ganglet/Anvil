from typing import List, Tuple
import torch
import torch.nn.functional as F

from models.base_model import BaseModel
from attacks.base_attack import BaseAttack, AdversarialExample


class PatchAttack(BaseAttack):
    """
    Adversarial patch — optimises a small square region to fool the model.
    Patch position comes from the Phase 2 saliency map: highest-saliency region.
    Simulates physical-world attacks (sticker on a stop sign, etc.).
    """

    def __init__(
        self,
        model: BaseModel,
        patch_size: int = 8,
        num_steps: int = 50,
        step_size: float = 0.05,
    ):
        super().__init__(model)
        self.patch_size = patch_size
        self.num_steps = num_steps
        self.step_size = step_size

    def _patch_position(self, profile: dict, h: int, w: int) -> Tuple[int, int]:
        """
        Use vulnerability_score as a proxy for saliency magnitude.
        In the absence of a full spatial saliency map in the profile, place the
        patch at the centre of the image — this is conservative and reproducible.
        A full spatial map would come from Phase 2's Captum output.
        """
        top = (h - self.patch_size) // 2
        left = (w - self.patch_size) // 2
        return top, left

    def attack(
        self,
        inputs: torch.Tensor,
        labels: List[int],
        profile: dict,
    ) -> List[AdversarialExample]:
        examples = []
        _, c, h, w = inputs.shape

        top, left = self._patch_position(profile, h, w)

        for i, label in enumerate(labels):
            x_orig = inputs[i].unsqueeze(0).detach()
            label_tensor = torch.tensor([label])

            # Initialise patch as random noise in [0, 1]
            patch = torch.rand(1, c, self.patch_size, self.patch_size, requires_grad=True)

            for _ in range(self.num_steps):
                x_patched = x_orig.clone()
                x_patched[:, :, top:top + self.patch_size, left:left + self.patch_size] = patch

                logits = self.model._model(x_patched)
                if hasattr(logits, "logits"):
                    logits = logits.logits

                loss = F.cross_entropy(logits, label_tensor)
                loss.backward()

                with torch.no_grad():
                    patch_data = (patch + self.step_size * patch.grad.sign()).clamp(0.0, 1.0)
                patch = patch_data.requires_grad_(True)

            with torch.no_grad():
                x_final = x_orig.clone()
                x_final[:, :, top:top + self.patch_size, left:left + self.patch_size] = patch

                orig_logits = self.model._model(x_orig)
                if hasattr(orig_logits, "logits"):
                    orig_logits = orig_logits.logits
                orig_pred = int(orig_logits.argmax(dim=1).item())

                adv_logits = self.model._model(x_final)
                if hasattr(adv_logits, "logits"):
                    adv_logits = adv_logits.logits
                adv_pred = int(adv_logits.argmax(dim=1).item())

            examples.append(
                AdversarialExample(
                    original_input=inputs[i].detach(),
                    perturbed_input=x_final.squeeze(0).detach(),
                    true_label=label,
                    original_pred=orig_pred,
                    adversarial_pred=adv_pred,
                    attack_name="PatchAttack",
                    attack_params={
                        "patch_size": self.patch_size,
                        "num_steps": self.num_steps,
                        "patch_top": top,
                        "patch_left": left,
                    },
                )
            )

        return examples
