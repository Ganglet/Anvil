import json
import os
import pytest
import torch

from models.image_model import ImageModel
from models.text_model import TextModel
from profiler.attack_surface import AttackSurfaceProfiler


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def image_model():
    return ImageModel(pretrained=True)


@pytest.fixture(scope="module")
def text_model():
    return TextModel()


@pytest.fixture(scope="module")
def image_inputs():
    torch.manual_seed(42)
    return torch.randn(4, 3, 224, 224)


@pytest.fixture(scope="module")
def image_targets(image_model, image_inputs):
    with torch.no_grad():
        logits = image_model.predict(image_inputs)
    return logits.argmax(dim=1).tolist()


@pytest.fixture(scope="module")
def text_inputs(text_model):
    sentences = [
        "This movie was absolutely wonderful",
        "Terrible film, waste of time",
        "An outstanding performance by the cast",
        "I did not enjoy this at all",
    ]
    return text_model.tokenize(sentences)["input_ids"]


@pytest.fixture(scope="module")
def text_targets(text_model, text_inputs):
    with torch.no_grad():
        logits = text_model.predict(text_inputs)
    return logits.argmax(dim=1).tolist()


# ── ImageModel profiler tests ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def image_profile(image_model, image_inputs, image_targets):
    profiler = AttackSurfaceProfiler(image_model)
    return profiler.profile(image_inputs, image_targets)


def test_image_profile_has_required_keys(image_profile):
    required = {
        "model", "num_samples", "gradient_norms",
        "activation_entropy", "mean_saliency_score",
        "attack_priority", "vulnerability_score",
    }
    assert required.issubset(image_profile.keys())


def test_image_profile_model_name(image_profile):
    assert image_profile["model"] == "resnet18"


def test_image_profile_num_samples(image_profile):
    assert image_profile["num_samples"] == 4


def test_image_gradient_norms_layers(image_profile):
    norms = image_profile["gradient_norms"]
    for layer in ["layer1", "layer2", "layer3", "layer4"]:
        assert layer in norms, f"Expected layer '{layer}' in gradient_norms"
        assert norms[layer] > 0, f"Gradient norm for {layer} should be > 0"


def test_image_activation_entropy_layers(image_profile):
    entropy = image_profile["activation_entropy"]
    for layer in ["layer1", "layer2", "layer3", "layer4"]:
        assert layer in entropy
        assert entropy[layer] >= 0


def test_image_attack_priority_is_ordered_list(image_profile):
    priority = image_profile["attack_priority"]
    assert isinstance(priority, list)
    assert len(priority) > 0
    valid_layers = {"layer1", "layer2", "layer3", "layer4", "avgpool"}
    assert all(layer in valid_layers for layer in priority)


def test_image_vulnerability_score_in_range(image_profile):
    score = image_profile["vulnerability_score"]
    assert 0.0 <= score <= 1.0


def test_image_saliency_score_positive(image_profile):
    assert image_profile["mean_saliency_score"] >= 0.0


# ── TextModel profiler tests ──────────────────────────────────────────────────

@pytest.fixture(scope="module")
def text_profile(text_model, text_inputs, text_targets):
    profiler = AttackSurfaceProfiler(text_model)
    return profiler.profile(text_inputs, text_targets)


def test_text_profile_has_required_keys(text_profile):
    required = {
        "model", "num_samples", "gradient_norms",
        "activation_entropy", "attack_priority", "vulnerability_score",
    }
    assert required.issubset(text_profile.keys())


def test_text_profile_model_name(text_profile):
    assert text_profile["model"] == "distilbert-sst2"


def test_text_gradient_norms_present(text_profile):
    norms = text_profile["gradient_norms"]
    assert len(norms) > 0
    for val in norms.values():
        assert val >= 0


def test_text_activation_entropy_present(text_profile):
    entropy = text_profile["activation_entropy"]
    assert len(entropy) > 0


def test_text_attack_priority_nonempty(text_profile):
    assert len(text_profile["attack_priority"]) > 0


def test_text_vulnerability_score_in_range(text_profile):
    score = text_profile["vulnerability_score"]
    assert 0.0 <= score <= 1.0


# ── Persistence tests ─────────────────────────────────────────────────────────

def test_save_and_load_profile(tmp_path, image_model, image_inputs, image_targets):
    profiler = AttackSurfaceProfiler(image_model)
    profile = profiler.profile(image_inputs, image_targets)

    path = str(tmp_path / "test_profile.json")
    profiler.save_profile(profile, path)

    assert os.path.exists(path)

    loaded = profiler.load_profile(path)
    assert loaded["model"] == profile["model"]
    assert loaded["vulnerability_score"] == profile["vulnerability_score"]
    assert loaded["attack_priority"] == profile["attack_priority"]
