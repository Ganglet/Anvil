from typing import List
import torch
import torch.nn.functional as F

from models.base_model import BaseModel
from attacks.base_attack import BaseAttack, AdversarialExample


class FGSMAttack(BaseAttack):
    """
    Fast Gradient Sign Method (Goodfellow et al., 2014).
    Single-step perturbation: x_adv = x + ε · sign(∇_x L(x, y))
    """

    def __init__(self, model: BaseModel, epsilon: float = 0.03):
        super().__init__(model)
        self.epsilon = epsilon

    def attack(
        self,
        inputs: torch.Tensor,
        labels: List[int],
        profile: dict,
    ) -> List[AdversarialExample]:
        examples = []

        for i, label in enumerate(labels):
            x = inputs[i].unsqueeze(0).clone().requires_grad_(True)

            logits = self.model._model(x)
            if hasattr(logits, "logits"):
                logits = logits.logits

            loss = F.cross_entropy(logits, torch.tensor([label]))
            loss.backward()

            x_adv = (x + self.epsilon * x.grad.sign()).detach().clamp(0.0, 1.0)

            orig_pred = int(logits.argmax(dim=1).item())

            with torch.no_grad():
                adv_logits = self.model._model(x_adv)
                if hasattr(adv_logits, "logits"):
                    adv_logits = adv_logits.logits
                adv_pred = int(adv_logits.argmax(dim=1).item())

            examples.append(
                AdversarialExample(
                    original_input=inputs[i].detach(),
                    perturbed_input=x_adv.squeeze(0),
                    true_label=label,
                    original_pred=orig_pred,
                    adversarial_pred=adv_pred,
                    attack_name="FGSM",
                    attack_params={"epsilon": self.epsilon},
                )
            )

        return examples
