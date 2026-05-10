from collections import Counter
from typing import List
import numpy as np
import umap
import hdbscan

from attacks.base_attack import AdversarialExample
from clustering.taxonomy import VulnerabilityCluster, VulnerabilityTaxonomy


class FailureModeClusterer:
    """
    Runs UMAP + HDBSCAN on feature vectors from Phase 4's FeatureExtractor.

    UMAP: reduces high-dimensional activation vectors to a low-dimensional
    embedding that preserves neighborhood structure (non-linear, unlike PCA).

    HDBSCAN: finds density-based clusters in that embedding without requiring
    a fixed cluster count. Points that don't fit any cluster get label -1 (noise).

    Output is a VulnerabilityTaxonomy — the input to Phase 5's LLM agent.
    """

    def __init__(self, n_components: int = 5, min_cluster_size: int = 2):
        self._n_components = n_components
        self._min_cluster_size = min_cluster_size

    def cluster(
        self,
        vectors: np.ndarray,
        examples: List[AdversarialExample],
        model_name: str = "unknown",
    ) -> VulnerabilityTaxonomy:
        """
        Args:
            vectors    — (N, D) feature matrix from FeatureExtractor
            examples   — the N AdversarialExample objects in the same order
            model_name — passed through to VulnerabilityTaxonomy for reporting

        Returns:
            VulnerabilityTaxonomy with named clusters + noise count.
        """
        n = len(vectors)

        if n < 2:
            return VulnerabilityTaxonomy(
                clusters=[],
                noise_count=n,
                total_failures=n,
                model_name=model_name,
            )

        embedding = self._umap_reduce(vectors, n)
        labels = self._hdbscan_label(embedding)
        clusters = self._build_clusters(labels, embedding, examples)
        noise_count = int((labels == -1).sum())

        return VulnerabilityTaxonomy(
            clusters=clusters,
            noise_count=noise_count,
            total_failures=n,
            model_name=model_name,
        )

    def _umap_reduce(self, vectors: np.ndarray, n: int) -> np.ndarray:
        n_neighbors = min(15, n - 1)
        n_components = min(self._n_components, n - 1)
        try:
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=42,
                low_memory=True,
            )
            return reducer.fit_transform(vectors)
        except Exception:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(2, n - 1))
            return pca.fit_transform(vectors)

    def _hdbscan_label(self, embedding: np.ndarray) -> np.ndarray:
        min_size = max(2, min(self._min_cluster_size, len(embedding) // 3))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size)
        return clusterer.fit_predict(embedding)

    def _build_clusters(
        self,
        labels: np.ndarray,
        embedding: np.ndarray,
        examples: List[AdversarialExample],
    ) -> List[VulnerabilityCluster]:
        cluster_ids = sorted(set(labels) - {-1})
        clusters = []

        for cid in cluster_ids:
            mask = labels == cid
            cluster_examples = [ex for ex, m in zip(examples, mask) if m]
            attack_counts = Counter(ex.attack_name for ex in cluster_examples)
            dominant = attack_counts.most_common(1)[0][0]
            centroid = embedding[mask].mean(axis=0)

            clusters.append(
                VulnerabilityCluster(
                    cluster_id=int(cid),
                    name=f"{dominant}_vulnerability_{cid}",
                    dominant_attack=dominant,
                    size=int(mask.sum()),
                    centroid=centroid,
                    examples=cluster_examples,
                    attack_distribution=dict(attack_counts),
                )
            )

        return clusters
