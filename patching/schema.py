from dataclasses import dataclass
from typing import List


@dataclass
class PatchResult:
    cluster_id: int
    cluster_name: str
    strategy: str
    safety_score: float
    passed: bool
    retries: int
    resistance_gain: float
    accuracy_drop: float


@dataclass
class PatchReport:
    model_name: str
    results: List[PatchResult]

    def summary(self) -> dict:
        return {
            "model": self.model_name,
            "total_clusters": len(self.results),
            "patched": sum(1 for r in self.results if r.passed),
            "unresolved": sum(1 for r in self.results if not r.passed),
            "clusters": [
                {
                    "id": r.cluster_id,
                    "name": r.cluster_name,
                    "strategy": r.strategy,
                    "safety_score": r.safety_score,
                    "passed": r.passed,
                    "retries": r.retries,
                    "resistance_gain": r.resistance_gain,
                    "accuracy_drop": r.accuracy_drop,
                }
                for r in self.results
            ],
        }
