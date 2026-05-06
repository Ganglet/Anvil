from dataclasses import dataclass, field
from typing import List


@dataclass
class ClusterExplanation:
    cluster_id: int
    cluster_name: str
    explanation: str
    patch_strategy: str
    patch_params: dict
    sources: List[str]


@dataclass
class ExplanationReport:
    model_name: str
    explanations: List[ClusterExplanation]

    def summary(self) -> dict:
        return {
            "model": self.model_name,
            "num_clusters_explained": len(self.explanations),
            "clusters": [
                {
                    "id": e.cluster_id,
                    "name": e.cluster_name,
                    "patch_strategy": e.patch_strategy,
                    "sources": e.sources,
                }
                for e in self.explanations
            ],
        }
