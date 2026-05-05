from typing import List, Tuple
import numpy as np
import torch

from models.base_model import BaseModel
from attacks.base_attack import AdversarialExample


class FeatureExtractor:
    """
    Extracts activation vectors from adversarial examples for clustering.

    Uses the top-ranked layer from Phase 2's attack_priority list.
    Pools spatial / sequence dimensions down to a 1D vector per example
    so UMAP has a consistent input shape regardless of model type.
    """

    def __init__(self, model: BaseModel, profile: dict):
        self._model = model
        priority = profile.get("attack_priority", [])
        if not priority:
            raise ValueError("Profile has empty attack_priority — run Phase 2 first.")
        self._layer = priority[0]
        self._is_text = "distilbert" in model.model_name

    def extract(
        self, examples: List[AdversarialExample]
    ) -> Tuple[np.ndarray, List[AdversarialExample]]:
        """
        Extract feature vectors for all *successful* adversarial examples.

        Returns:
            vectors  — (N, D) float32 array, one row per successful example
            filtered — the N AdversarialExample objects that produced those rows
        """
        successful = [e for e in examples if e.success]
        if not successful:
            return np.empty((0,), dtype=np.float32), []

        vectors = []
        for ex in successful:
            inp = ex.perturbed_input.unsqueeze(0)
            acts = self._model.get_activations(inp, self._layer)
            vectors.append(self._pool(acts))

        return np.stack(vectors).astype(np.float32), successful

    def _pool(self, acts: torch.Tensor) -> np.ndarray:
        # Image activations: (1, C, H, W) → mean over spatial → (C,)
        # Text activations:  (1, seq, D)  → mean over sequence → (D,)
        if acts.dim() == 4:
            return acts.mean(dim=[2, 3]).squeeze(0).detach().numpy()
        elif acts.dim() == 3:
            return acts.mean(dim=1).squeeze(0).detach().numpy()
        else:
            return acts.squeeze(0).detach().numpy()
