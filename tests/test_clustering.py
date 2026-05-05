import numpy as np
import pytest
import torch

from attacks.base_attack import AdversarialExample
from clustering.feature_extractor import FeatureExtractor
from clustering.clusterer import FailureModeClusterer
from clustering.taxonomy import VulnerabilityCluster, VulnerabilityTaxonomy
from models.image_model import ImageModel


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def image_model():
    return ImageModel(pretrained=False, num_classes=10)


@pytest.fixture
def fake_profile():
    return {
        "model": "resnet18",
        "attack_priority": ["layer4", "layer3", "layer2", "layer1", "avgpool"],
        "gradient_norms": {"layer4": 1.2, "layer3": 0.8},
        "activation_entropy": {"layer4": 3.1, "layer3": 4.2},
        "vulnerability_score": 0.72,
    }


def _make_example(success: bool, attack_name: str = "FGSM") -> AdversarialExample:
    """Creates an AdversarialExample with a real (3,224,224) tensor."""
    x = torch.rand(3, 224, 224)
    return AdversarialExample(
        original_input=x,
        perturbed_input=x + 0.01 * torch.randn_like(x),
        true_label=0,
        original_pred=0,
        adversarial_pred=1 if success else 0,
        attack_name=attack_name,
    )


def _make_vectors(n: int, n_clusters: int = 2, dim: int = 32) -> np.ndarray:
    """Synthetic clustered vectors — useful for testing clusterer without a model."""
    rng = np.random.default_rng(42)
    vecs = []
    for i in range(n_clusters):
        center = rng.standard_normal(dim) * 10
        cluster = center + rng.standard_normal((n // n_clusters, dim)) * 0.5
        vecs.append(cluster)
    return np.vstack(vecs).astype(np.float32)


# ── FeatureExtractor ──────────────────────────────────────────────────────────

def test_extractor_returns_correct_shape(image_model, fake_profile):
    examples = [_make_example(success=True) for _ in range(4)]
    extractor = FeatureExtractor(image_model, fake_profile)
    vectors, filtered = extractor.extract(examples)
    assert vectors.shape[0] == 4
    assert vectors.ndim == 2
    assert len(filtered) == 4


def test_extractor_filters_unsuccessful(image_model, fake_profile):
    examples = [
        _make_example(success=True),
        _make_example(success=False),
        _make_example(success=True),
        _make_example(success=False),
    ]
    extractor = FeatureExtractor(image_model, fake_profile)
    vectors, filtered = extractor.extract(examples)
    assert vectors.shape[0] == 2
    assert len(filtered) == 2
    assert all(e.success for e in filtered)


def test_extractor_all_failures_returns_empty(image_model, fake_profile):
    examples = [_make_example(success=False) for _ in range(3)]
    extractor = FeatureExtractor(image_model, fake_profile)
    vectors, filtered = extractor.extract(examples)
    assert vectors.shape[0] == 0
    assert filtered == []


def test_extractor_empty_list(image_model, fake_profile):
    extractor = FeatureExtractor(image_model, fake_profile)
    vectors, filtered = extractor.extract([])
    assert filtered == []


def test_extractor_requires_attack_priority(image_model):
    with pytest.raises(ValueError, match="attack_priority"):
        FeatureExtractor(image_model, {"attack_priority": []})


def test_extractor_output_dtype(image_model, fake_profile):
    examples = [_make_example(success=True)]
    extractor = FeatureExtractor(image_model, fake_profile)
    vectors, _ = extractor.extract(examples)
    assert vectors.dtype == np.float32


# ── FailureModeClusterer ──────────────────────────────────────────────────────

def _make_examples_for_vectors(n: int, attack_name: str = "FGSM") -> list:
    return [_make_example(success=True, attack_name=attack_name) for _ in range(n)]


def test_clusterer_produces_taxonomy():
    n = 20
    vectors = _make_vectors(n, n_clusters=2)
    examples = _make_examples_for_vectors(n)
    clusterer = FailureModeClusterer(n_components=2, min_cluster_size=2)
    taxonomy = clusterer.cluster(vectors, examples, model_name="resnet18")
    assert isinstance(taxonomy, VulnerabilityTaxonomy)
    assert taxonomy.total_failures == n
    assert taxonomy.noise_count + sum(c.size for c in taxonomy.clusters) == n


def test_clusterer_cluster_names_contain_attack():
    n = 20
    vectors = _make_vectors(n, n_clusters=2)
    examples = _make_examples_for_vectors(n, attack_name="PGD")
    clusterer = FailureModeClusterer(n_components=2, min_cluster_size=2)
    taxonomy = clusterer.cluster(vectors, examples, model_name="resnet18")
    for cluster in taxonomy.clusters:
        assert "PGD" in cluster.name


def test_clusterer_cluster_dominant_attack():
    n = 18
    vectors = _make_vectors(n, n_clusters=2)
    examples = (
        _make_examples_for_vectors(9, attack_name="FGSM")
        + _make_examples_for_vectors(9, attack_name="Patch")
    )
    clusterer = FailureModeClusterer(n_components=2, min_cluster_size=2)
    taxonomy = clusterer.cluster(vectors, examples, model_name="resnet18")
    dominant_attacks = {c.dominant_attack for c in taxonomy.clusters}
    assert dominant_attacks.issubset({"FGSM", "Patch"})


def test_clusterer_centroid_shape():
    n = 20
    vectors = _make_vectors(n, n_clusters=2)
    examples = _make_examples_for_vectors(n)
    clusterer = FailureModeClusterer(n_components=2, min_cluster_size=2)
    taxonomy = clusterer.cluster(vectors, examples)
    for cluster in taxonomy.clusters:
        assert cluster.centroid.ndim == 1
        assert cluster.centroid.shape[0] == 2  # n_components=2


def test_clusterer_too_few_examples_returns_empty_taxonomy():
    vectors = np.array([[1.0, 2.0]], dtype=np.float32)
    examples = _make_examples_for_vectors(1)
    clusterer = FailureModeClusterer()
    taxonomy = clusterer.cluster(vectors, examples)
    assert taxonomy.clusters == []
    assert taxonomy.noise_count == 1
    assert taxonomy.total_failures == 1


def test_clusterer_zero_examples():
    clusterer = FailureModeClusterer()
    taxonomy = clusterer.cluster(np.empty((0, 32), dtype=np.float32), [])
    assert taxonomy.clusters == []
    assert taxonomy.total_failures == 0


# ── VulnerabilityTaxonomy ─────────────────────────────────────────────────────

def _make_taxonomy() -> VulnerabilityTaxonomy:
    cluster = VulnerabilityCluster(
        cluster_id=0,
        name="FGSM_vulnerability_0",
        dominant_attack="FGSM",
        size=5,
        centroid=np.array([0.1, 0.2]),
        examples=[_make_example(success=True)],
        attack_distribution={"FGSM": 4, "PGD": 1},
    )
    return VulnerabilityTaxonomy(
        clusters=[cluster],
        noise_count=2,
        total_failures=7,
        model_name="resnet18",
    )


def test_taxonomy_summary_keys():
    taxonomy = _make_taxonomy()
    s = taxonomy.summary()
    assert set(s.keys()) == {"model", "total_failures", "num_clusters", "noise_count", "clusters"}


def test_taxonomy_summary_cluster_keys():
    taxonomy = _make_taxonomy()
    s = taxonomy.summary()
    assert len(s["clusters"]) == 1
    assert set(s["clusters"][0].keys()) == {"id", "name", "size", "dominant_attack", "attack_distribution"}


def test_taxonomy_summary_counts():
    taxonomy = _make_taxonomy()
    s = taxonomy.summary()
    assert s["total_failures"] == 7
    assert s["num_clusters"] == 1
    assert s["noise_count"] == 2


def test_taxonomy_empty_clusters():
    taxonomy = VulnerabilityTaxonomy(
        clusters=[], noise_count=3, total_failures=3, model_name="resnet18"
    )
    s = taxonomy.summary()
    assert s["num_clusters"] == 0
    assert s["clusters"] == []
