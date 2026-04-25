import pytest
import torch
from models.image_model import ImageModel
from models.text_model import TextModel


# ── ImageModel ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def image_model():
    return ImageModel(pretrained=True)


@pytest.fixture(scope="module")
def dummy_image():
    # (1, C, H, W) — single ImageNet-normalized image
    return torch.randn(1, 3, 224, 224)


def test_image_predict_shape(image_model, dummy_image):
    logits = image_model.predict(dummy_image)
    assert logits.shape == (1, 1000), f"Expected (1, 1000), got {logits.shape}"


def test_image_predict_no_grad_leak(image_model, dummy_image):
    logits = image_model.predict(dummy_image)
    assert not logits.requires_grad


def test_image_gradients_shape(image_model, dummy_image):
    grads = image_model.get_gradients(dummy_image, target_class=0)
    assert grads.shape == dummy_image.shape, (
        f"Gradient shape {grads.shape} != input shape {dummy_image.shape}"
    )


def test_image_gradients_not_zero(image_model, dummy_image):
    grads = image_model.get_gradients(dummy_image, target_class=0)
    assert grads.abs().sum() > 0, "Gradients are all zero — something is wrong"


def test_image_activations_layer4(image_model, dummy_image):
    # layer4 is the penultimate block — what Phase 4 clustering will use
    acts = image_model.get_activations(dummy_image, layer_name="layer4")
    assert acts.shape[0] == 1
    assert acts.ndim == 4  # (batch, channels, h, w)


def test_image_activations_avgpool(image_model, dummy_image):
    # avgpool collapses spatial dims → (batch, 512, 1, 1), good for clustering
    acts = image_model.get_activations(dummy_image, layer_name="avgpool")
    assert acts.shape == (1, 512, 1, 1)


def test_image_available_layers(image_model):
    layers = image_model.get_available_layers()
    assert "layer4" in layers
    assert "avgpool" in layers
    assert len(layers) > 10


# ── TextModel ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def text_model():
    return TextModel()


@pytest.fixture(scope="module")
def dummy_tokens(text_model):
    return text_model.tokenize(["This film was absolutely fantastic!"])["input_ids"]


def test_text_predict_shape(text_model, dummy_tokens):
    logits = text_model.predict(dummy_tokens)
    assert logits.shape == (1, 2), f"Expected (1, 2), got {logits.shape}"


def test_text_predict_from_text(text_model):
    logits = text_model.predict_from_text(["great movie", "terrible film"])
    assert logits.shape == (2, 2)


def test_text_gradients_shape(text_model, dummy_tokens):
    grads = text_model.get_gradients(dummy_tokens, target_class=1)
    # shape: (batch, seq_len, hidden_size=768)
    assert grads.ndim == 3
    assert grads.shape[0] == 1
    assert grads.shape[2] == 768


def test_text_gradients_not_zero(text_model, dummy_tokens):
    grads = text_model.get_gradients(dummy_tokens, target_class=1)
    assert grads.abs().sum() > 0


def test_text_activations_last_transformer(text_model, dummy_tokens):
    # transformer.layer.5 is the final transformer block — used for clustering
    acts = text_model.get_activations(
        dummy_tokens, layer_name="distilbert.transformer.layer.5"
    )
    assert acts.shape[0] == 1
    assert acts.ndim == 3  # (batch, seq_len, hidden_size)


def test_text_available_layers(text_model):
    layers = text_model.get_available_layers()
    assert any("transformer" in l for l in layers)
    assert len(layers) > 5
