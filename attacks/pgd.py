from typing import List
import torch
import torch.nn.functional as F

from models.base_model import BaseModel
from attacks.base_attack import BaseAttack, AdversarialExample


class PGDAttack(BaseAttack):
    """
    Projected Gradient Descent (Madry et al., 2018).
    Iterative FGSM with projection back into the ε-ball after each step.
    Strictly stronger than FGSM — finds deeper failure modes.
    """

    def __init__(
        self,
        model: BaseModel,
        epsilon: float = 0.03,
        step_size: float = 0.007,
        num_steps: int = 40,
    ):
        super().__init__(model)
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps

    def attack(
        self,
        inputs: torch.Tensor,
        labels: List[int],
        profile: dict,
    ) -> List[AdversarialExample]:
        examples = []

        for i, label in enumerate(labels):
            x_orig = inputs[i].unsqueeze(0).detach()
            # Random start inside the ε-ball
            x_adv = x_orig + torch.empty_like(x_orig).uniform_(-self.epsilon, self.epsilon)
            x_adv = x_adv.clamp(0.0, 1.0)

            label_tensor = torch.tensor([label])

            for _ in range(self.num_steps):
                x_adv = x_adv.requires_grad_(True)

                logits = self.model._model(x_adv)
                if hasattr(logits, "logits"):
                    logits = logits.logits

                loss = F.cross_entropy(logits, label_tensor)
                loss.backward()

                with torch.no_grad():
                    x_adv = x_adv + self.step_size * x_adv.grad.sign()
                    # Project back into ε-ball around original
                    delta = (x_adv - x_orig).clamp(-self.epsilon, self.epsilon)
                    x_adv = (x_orig + delta).clamp(0.0, 1.0)

            with torch.no_grad():
                orig_logits = self.model._model(x_orig)
                if hasattr(orig_logits, "logits"):
                    orig_logits = orig_logits.logits
                orig_pred = int(orig_logits.argmax(dim=1).item())

                adv_logits = self.model._model(x_adv)
                if hasattr(adv_logits, "logits"):
                    adv_logits = adv_logits.logits
                adv_pred = int(adv_logits.argmax(dim=1).item())

            examples.append(
                AdversarialExample(
                    original_input=inputs[i].detach(),
                    perturbed_input=x_adv.squeeze(0).detach(),
                    true_label=label,
                    original_pred=orig_pred,
                    adversarial_pred=adv_pred,
                    attack_name="PGD",
                    attack_params={
                        "epsilon": self.epsilon,
                        "step_size": self.step_size,
                        "num_steps": self.num_steps,
                    },
                )
            )

        return examples
