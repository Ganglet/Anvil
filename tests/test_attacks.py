import pytest
import torch

from models.image_model import ImageModel
from models.text_model import TextModel
from attacks.fgsm import FGSMAttack
from attacks.pgd import PGDAttack
from attacks.patch_attack import PatchAttack
from attacks.semantic_attack import SemanticAttack
from attacks.text_attack import TextAttack
from attacks.engine import AttackEngine
from attacks.base_attack import AdversarialExample


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def image_model():
    return ImageModel(pretrained=False, num_classes=10)


@pytest.fixture(scope="module")
def text_model():
    return TextModel()


@pytest.fixture(scope="module")
def image_inputs():
    torch.manual_seed(0)
    return torch.rand(2, 3, 224, 224)


@pytest.fixture(scope="module")
def text_inputs(text_model):
    sentences = ["this movie was great", "terrible acting"]
    enc = text_model._tokenizer(
        sentences, return_tensors="pt", truncation=True,
        padding="max_length", max_length=32
    )
    return enc["input_ids"]


@pytest.fixture(scope="module")
def image_labels():
    return [3, 7]


@pytest.fixture(scope="module")
def text_labels():
    return [1, 0]


@pytest.fixture(scope="module")
def dummy_profile():
    return {
        "model": "resnet18",
        "vulnerability_score": 0.65,
        "attack_priority": ["layer4", "layer3", "layer2", "layer1", "avgpool"],
        "gradient_norms": {"layer1": 0.2, "layer2": 0.4, "layer3": 0.6, "layer4": 0.9, "avgpool": 0.1},
        "activation_entropy": {"layer1": 3.5, "layer2": 3.0, "layer3": 2.5, "layer4": 1.8, "avgpool": 4.0},
        "mean_saliency_score": 0.05,
    }


# ── AdversarialExample ────────────────────────────────────────────────────────

def test_adversarial_example_success_property():
    orig = torch.zeros(3, 4, 4)
    adv = torch.ones(3, 4, 4)
    ex = AdversarialExample(orig, adv, true_label=0, original_pred=0, adversarial_pred=1,
                            attack_name="test")
    assert ex.success is True


def test_adversarial_example_failure_property():
    orig = torch.zeros(3, 4, 4)
    ex = AdversarialExample(orig, orig, true_label=0, original_pred=0, adversarial_pred=0,
                            attack_name="test")
    assert ex.success is False


# ── FGSM ─────────────────────────────────────────────────────────────────────

def test_fgsm_returns_correct_count(image_model, image_inputs, image_labels, dummy_profile):
    attack = FGSMAttack(image_model, epsilon=0.03)
    examples = attack.attack(image_inputs, image_labels, dummy_profile)
    assert len(examples) == len(image_labels)


def test_fgsm_output_shape(image_model, image_inputs, image_labels, dummy_profile):
    attack = FGSMAttack(image_model, epsilon=0.03)
    examples = attack.attack(image_inputs, image_labels, dummy_profile)
    for ex in examples:
        assert ex.perturbed_input.shape == image_inputs[0].shape


def test_fgsm_perturbation_bounded(image_model, image_inputs, image_labels, dummy_profile):
    eps = 0.03
    attack = FGSMAttack(image_model, epsilon=eps)
    examples = attack.attack(image_inputs, image_labels, dummy_profile)
    for ex in examples:
        delta = (ex.perturbed_input - ex.original_input).abs()
        assert delta.max().item() <= eps + 1e-5


def test_fgsm_attack_name(image_model, image_inputs, image_labels, dummy_profile):
    attack = FGSMAttack(image_model, epsilon=0.03)
    examples = attack.attack(image_inputs, image_labels, dummy_profile)
    assert all(ex.attack_name == "FGSM" for ex in examples)


def test_fgsm_output_in_valid_range(image_model, image_inputs, image_labels, dummy_profile):
    attack = FGSMAttack(image_model, epsilon=0.03)
    examples = attack.attack(image_inputs, image_labels, dummy_profile)
    for ex in examples:
        assert ex.perturbed_input.min().item() >= -1e-5
        assert ex.perturbed_input.max().item() <= 1.0 + 1e-5


# ── PGD ──────────────────────────────────────────────────────────────────────

def test_pgd_returns_correct_count(image_model, image_inputs, image_labels, dummy_profile):
    attack = PGDAttack(image_model, epsilon=0.03, num_steps=5)
    examples = attack.attack(image_inputs, image_labels, dummy_profile)
    assert len(examples) == len(image_labels)


def test_pgd_perturbation_bounded(image_model, image_inputs, image_labels, dummy_profile):
    eps = 0.03
    attack = PGDAttack(image_model, epsilon=eps, num_steps=5)
    examples = attack.attack(image_inputs, image_labels, dummy_profile)
    for ex in examples:
        delta = (ex.perturbed_input - ex.original_input).abs()
        assert delta.max().item() <= eps + 1e-4


def test_pgd_attack_name(image_model, image_inputs, image_labels, dummy_profile):
    attack = PGDAttack(image_model, num_steps=5)
    examples = attack.attack(image_inputs, image_labels, dummy_profile)
    assert all(ex.attack_name == "PGD" for ex in examples)


# ── PatchAttack ───────────────────────────────────────────────────────────────

def test_patch_returns_correct_count(image_model, image_inputs, image_labels, dummy_profile):
    attack = PatchAttack(image_model, patch_size=8, num_steps=5)
    examples = attack.attack(image_inputs, image_labels, dummy_profile)
    assert len(examples) == len(image_labels)


def test_patch_output_shape(image_model, image_inputs, image_labels, dummy_profile):
    attack = PatchAttack(image_model, patch_size=8, num_steps=5)
    examples = attack.attack(image_inputs, image_labels, dummy_profile)
    for ex in examples:
        assert ex.perturbed_input.shape == image_inputs[0].shape


def test_patch_attack_name(image_model, image_inputs, image_labels, dummy_profile):
    attack = PatchAttack(image_model, num_steps=5)
    examples = attack.attack(image_inputs, image_labels, dummy_profile)
    assert all(ex.attack_name == "PatchAttack" for ex in examples)


# ── SemanticAttack ────────────────────────────────────────────────────────────

def test_semantic_returns_correct_count(image_model, image_inputs, image_labels, dummy_profile):
    attack = SemanticAttack(image_model)
    examples = attack.attack(image_inputs, image_labels, dummy_profile)
    assert len(examples) == len(image_labels)


def test_semantic_output_shape(image_model, image_inputs, image_labels, dummy_profile):
    attack = SemanticAttack(image_model)
    examples = attack.attack(image_inputs, image_labels, dummy_profile)
    for ex in examples:
        assert ex.perturbed_input.shape == image_inputs[0].shape


def test_semantic_attack_name(image_model, image_inputs, image_labels, dummy_profile):
    attack = SemanticAttack(image_model)
    examples = attack.attack(image_inputs, image_labels, dummy_profile)
    assert all(ex.attack_name == "SemanticAttack" for ex in examples)


def test_semantic_records_transform_params(image_model, image_inputs, image_labels, dummy_profile):
    attack = SemanticAttack(image_model)
    examples = attack.attack(image_inputs, image_labels, dummy_profile)
    for ex in examples:
        assert "transform" in ex.attack_params
        assert "value" in ex.attack_params


# ── TextAttack ────────────────────────────────────────────────────────────────

def test_text_attack_returns_correct_count(text_model, text_inputs, text_labels, dummy_profile):
    attack = TextAttack(text_model)
    examples = attack.attack(text_inputs, text_labels, dummy_profile)
    assert len(examples) == len(text_labels)


def test_text_attack_name(text_model, text_inputs, text_labels, dummy_profile):
    attack = TextAttack(text_model)
    examples = attack.attack(text_inputs, text_labels, dummy_profile)
    assert all(ex.attack_name == "TextAttack" for ex in examples)


def test_text_attack_records_texts(text_model, text_inputs, text_labels, dummy_profile):
    attack = TextAttack(text_model)
    examples = attack.attack(text_inputs, text_labels, dummy_profile)
    for ex in examples:
        assert "original_text" in ex.attack_params
        assert "perturbed_text" in ex.attack_params


def test_text_attack_changes_text(text_model, text_inputs, text_labels, dummy_profile):
    attack = TextAttack(text_model)
    examples = attack.attack(text_inputs, text_labels, dummy_profile)
    for ex in examples:
        # Perturbation should modify at least something
        assert ex.attack_params["original_text"] != ex.attack_params["perturbed_text"] or \
               len(ex.attack_params["original_text"].split()) > 0


# ── AttackEngine ──────────────────────────────────────────────────────────────

def test_engine_image_returns_all_strategies(image_model, image_inputs, image_labels, dummy_profile):
    engine = AttackEngine(image_model)
    results = engine.run(image_inputs, image_labels, dummy_profile)
    assert set(results.keys()) == {"FGSM", "PGD", "Patch", "Semantic"}


def test_engine_text_returns_text_strategy(text_model, text_inputs, text_labels, dummy_profile):
    engine = AttackEngine(text_model)
    results = engine.run(text_inputs, text_labels, dummy_profile)
    assert "Text" in results


def test_engine_success_rate_in_range(image_model, image_inputs, image_labels, dummy_profile):
    engine = AttackEngine(image_model)
    results = engine.run(image_inputs, image_labels, dummy_profile)
    rates = engine.success_rate(results)
    for name, rate in rates.items():
        assert 0.0 <= rate <= 1.0, f"{name} success rate {rate} out of range"


def test_engine_each_strategy_has_right_count(image_model, image_inputs, image_labels, dummy_profile):
    engine = AttackEngine(image_model)
    results = engine.run(image_inputs, image_labels, dummy_profile)
    for name, examples in results.items():
        assert len(examples) == len(image_labels), f"{name} has wrong example count"
