from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np

from attacks.base_attack import AdversarialExample


@dataclass
class VulnerabilityCluster:
    cluster_id: int
    name: str
    dominant_attack: str
    size: int
    centroid: np.ndarray
    examples: List[AdversarialExample]
    attack_distribution: Dict[str, int]


@dataclass
class VulnerabilityTaxonomy:
    clusters: List[VulnerabilityCluster]
    noise_count: int
    total_failures: int
    model_name: str

    def summary(self) -> dict:
        return {
            "model": self.model_name,
            "total_failures": self.total_failures,
            "num_clusters": len(self.clusters),
            "noise_count": self.noise_count,
            "clusters": [
                {
                    "id": c.cluster_id,
                    "name": c.name,
                    "size": c.size,
                    "dominant_attack": c.dominant_attack,
                    "attack_distribution": c.attack_distribution,
                }
                for c in self.clusters
            ],
        }
