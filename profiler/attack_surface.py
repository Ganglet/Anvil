import json
from typing import Dict, List, Optional
import torch
import numpy as np
from captum.attr import Saliency

from models.base_model import BaseModel


class AttackSurfaceProfiler:
    """
    Profiles a model's attack surface on a set of clean inputs.
    Produces a JSON dict that Phase 3 uses to prioritise attack targets.

    Three measurements:
      - Gradient norms per layer  → which layers amplify input changes most
      - Activation entropy        → how concentrated/brittle each layer is
      - Input saliency score      → how sensitive the output is to raw input
    """

    # Layers profiled for ResNet-18 and DistilBERT respectively
    IMAGE_PROFILE_LAYERS = ["layer1", "layer2", "layer3", "layer4", "avgpool"]
    TEXT_PROFILE_LAYERS = [
        "distilbert.transformer.layer.0",
        "distilbert.transformer.layer.1",
        "distilbert.transformer.layer.2",
        "distilbert.transformer.layer.3",
        "distilbert.transformer.layer.4",
        "distilbert.transformer.layer.5",
    ]

    def __init__(self, model: BaseModel):
        self.model = model
        self._profile_layers = (
            self.TEXT_PROFILE_LAYERS
            if "distilbert" in model.model_name
            else self.IMAGE_PROFILE_LAYERS
        )

    def profile(
        self,
        inputs: torch.Tensor,
        target_classes: List[int],
    ) -> Dict:
        """
        Run full profiling pass on inputs.

        Args:
            inputs:         (N, C, H, W) for images or (N, seq_len) for text
            target_classes: predicted or ground-truth class per sample

        Returns:
            Profile dict — save with save_profile() or pass directly to Phase 3.
        """
        gradient_norms = self._compute_gradient_norms(inputs, target_classes)
        activation_entropy = self._compute_activation_entropy(inputs)
        mean_saliency = self._compute_saliency(inputs, target_classes)
        attack_priority = self._rank_layers(gradient_norms, activation_entropy)
        vulnerability_score = self._overall_vulnerability(
            gradient_norms, activation_entropy, mean_saliency
        )

        return {
            "model": self.model.model_name,
            "num_samples": len(inputs),
            "gradient_norms": gradient_norms,
            "activation_entropy": activation_entropy,
            "mean_saliency_score": round(float(mean_saliency), 4),
            "attack_priority": attack_priority,
            "vulnerability_score": round(float(vulnerability_score), 4),
        }

    # ── Private methods ───────────────────────────────────────────────────────

    def _compute_gradient_norms(
        self, inputs: torch.Tensor, target_classes: List[int]
    ) -> Dict[str, float]:
        """
        L2 norm of the gradient of the target class logit w.r.t. each input.
        Averaged across all samples. Higher = layer amplifies perturbations more.
        """
        norms: Dict[str, List[float]] = {l: [] for l in self._profile_layers}

        for i, target in enumerate(target_classes):
            sample = inputs[i].unsqueeze(0)
            grads = self.model.get_gradients(sample, target_class=target)
            norm = float(grads.norm().item())
            # input-level gradient applies to all layers as a baseline signal
            for layer in self._profile_layers:
                norms[layer].append(norm)

        # Per-layer activation gradients via hooks
        for layer_name in self._profile_layers:
            layer_norms = []
            for i, target in enumerate(target_classes):
                sample = inputs[i].unsqueeze(0)
                try:
                    acts = self.model.get_activations(sample, layer_name)
                    layer_norms.append(float(acts.norm().item()))
                except (ValueError, KeyError):
                    pass
            if layer_norms:
                norms[layer_name] = layer_norms

        return {
            layer: round(float(np.mean(vals)), 4)
            for layer, vals in norms.items()
            if vals
        }

    def _compute_activation_entropy(
        self, inputs: torch.Tensor
    ) -> Dict[str, float]:
        """
        Shannon entropy of the activation distribution at each layer, averaged
        across samples. Low entropy = concentrated activations = brittle layer.
        """
        entropy_scores: Dict[str, List[float]] = {}

        for layer_name in self._profile_layers:
            entropies = []
            for i in range(len(inputs)):
                sample = inputs[i].unsqueeze(0)
                try:
                    acts = self.model.get_activations(sample, layer_name)
                    flat = acts.flatten().detach().numpy()
                    # Normalise to a probability distribution via softmax-like transform
                    flat = np.abs(flat)
                    total = flat.sum()
                    if total > 0:
                        flat = flat / total
                        # Shannon entropy: -sum(p * log(p))
                        flat = flat[flat > 0]
                        entropy = float(-np.sum(flat * np.log(flat)))
                        entropies.append(entropy)
                except (ValueError, KeyError):
                    pass
            if entropies:
                entropy_scores[layer_name] = round(float(np.mean(entropies)), 4)

        return entropy_scores

    def _compute_saliency(
        self, inputs: torch.Tensor, target_classes: List[int]
    ) -> float:
        """
        Mean absolute saliency across all inputs using Captum.
        Returns a scalar: average influence of individual input features on output.
        Only supported for image models (requires float input).
        For text models returns 0.0 — text saliency is handled via gradient norms.
        """
        if "distilbert" in self.model.model_name:
            return 0.0

        saliency = Saliency(self.model._model)
        scores = []

        for i, target in enumerate(target_classes):
            sample = inputs[i].unsqueeze(0).requires_grad_(True)
            try:
                attrs = saliency.attribute(sample, target=target)
                scores.append(float(attrs.abs().mean().item()))
            except Exception:
                pass

        return float(np.mean(scores)) if scores else 0.0

    def _rank_layers(
        self,
        gradient_norms: Dict[str, float],
        activation_entropy: Dict[str, float],
    ) -> List[str]:
        """
        Rank layers by vulnerability: higher gradient norm + lower entropy = more
        vulnerable. Phase 3 targets layers at the top of this list first.
        """
        if not gradient_norms:
            return []

        # Normalise each metric to [0, 1]
        norms = np.array(list(gradient_norms.values()))
        entropies = np.array(
            [activation_entropy.get(l, 0.0) for l in gradient_norms]
        )

        norm_range = norms.max() - norms.min()
        entr_range = entropies.max() - entropies.min()

        norm_norm = (norms - norms.min()) / norm_range if norm_range > 0 else norms
        # Invert entropy: low entropy = high vulnerability
        entr_inv = (
            1 - (entropies - entropies.min()) / entr_range
            if entr_range > 0
            else 1 - entropies
        )

        scores = 0.6 * norm_norm + 0.4 * entr_inv
        ranked = sorted(
            zip(gradient_norms.keys(), scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return [layer for layer, _ in ranked]

    def _overall_vulnerability(
        self,
        gradient_norms: Dict[str, float],
        activation_entropy: Dict[str, float],
        mean_saliency: float,
    ) -> float:
        """
        Single scalar vulnerability score in [0, 1].
        Combines mean gradient norm (normalised), mean inverse entropy, and saliency.
        """
        if not gradient_norms:
            return 0.0

        norm_vals = list(gradient_norms.values())
        entr_vals = list(activation_entropy.values())

        max_norm = max(norm_vals) if norm_vals else 1.0
        mean_norm_score = np.mean(norm_vals) / max_norm if max_norm > 0 else 0.0

        max_entr = max(entr_vals) if entr_vals else 1.0
        mean_entr_score = 1 - (np.mean(entr_vals) / max_entr) if max_entr > 0 else 0.0

        score = 0.5 * mean_norm_score + 0.3 * mean_entr_score + 0.2 * mean_saliency
        return float(np.clip(score, 0.0, 1.0))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_profile(self, profile: Dict, output_path: str) -> None:
        with open(output_path, "w") as f:
            json.dump(profile, f, indent=2)

    def load_profile(self, path: str) -> Dict:
        with open(path, "r") as f:
            return json.load(f)
