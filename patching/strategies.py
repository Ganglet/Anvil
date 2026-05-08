from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel
from clustering.taxonomy import VulnerabilityCluster

_LR = {"high": 1e-4, "medium": 5e-5, "low": 1e-5}
_EPSILON = {"high": 0.03, "medium": 0.02, "low": 0.01}


def _get_optimizer(model: BaseModel, strength: str) -> torch.optim.Optimizer:
    lr = _LR[strength]
    if hasattr(model._model, "fc"):
        params = list(model._model.fc.parameters())
    elif hasattr(model._model, "classifier"):
        params = list(model._model.classifier.parameters())
    else:
        params = list(model._model.parameters())
    return torch.optim.Adam(params, lr=lr)


def _fine_tune_step(
    model: BaseModel,
    inputs: torch.Tensor,
    labels: List[int],
    optimizer: torch.optim.Optimizer,
    steps: int,
):
    model._model.train()
    label_tensor = torch.tensor(labels)
    for _ in range(steps):
        optimizer.zero_grad()
        logits = model._model(inputs)
        if hasattr(logits, "logits"):
            logits = logits.logits
        loss = F.cross_entropy(logits, label_tensor)
        loss.backward()
        optimizer.step()
    model._model.eval()


def adversarial_training(
    model: BaseModel,
    cluster: VulnerabilityCluster,
    clean_inputs: torch.Tensor,
    clean_labels: List[int],
    strength: str,
    steps: int,
):
    """Fine-tune on PGD adversarial examples mixed with clean data."""
    optimizer = _get_optimizer(model, strength)

    if "distilbert" in model.model_name:
        _fine_tune_step(model, clean_inputs, clean_labels, optimizer, steps)
        return

    eps = _EPSILON[strength]
    step_size = eps / 4
    pgd_steps = max(2, steps // 2)

    x_adv = clean_inputs.clone().detach()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0.0, 1.0)

    label_tensor = torch.tensor(clean_labels)
    model._model.eval()

    for _ in range(pgd_steps):
        x_adv = x_adv.requires_grad_(True)
        model._model.zero_grad()
        logits = model._model(x_adv)
        if hasattr(logits, "logits"):
            logits = logits.logits
        loss = F.cross_entropy(logits, label_tensor)
        loss.backward()
        with torch.no_grad():
            x_adv = x_adv + step_size * x_adv.grad.sign()
            delta = (x_adv - clean_inputs).clamp(-eps, eps)
            x_adv = (clean_inputs + delta).clamp(0.0, 1.0)

    mixed_inputs = torch.cat([clean_inputs, x_adv.detach()], dim=0)
    mixed_labels = list(clean_labels) + list(clean_labels)
    _fine_tune_step(model, mixed_inputs, mixed_labels, optimizer, steps)


def stylized_augmentation(
    model: BaseModel,
    cluster: VulnerabilityCluster,
    clean_inputs: torch.Tensor,
    clean_labels: List[int],
    strength: str,
    steps: int,
):
    """Fine-tune on texture-randomized inputs to reduce texture bias."""
    optimizer = _get_optimizer(model, strength)

    if "distilbert" in model.model_name:
        _fine_tune_step(model, clean_inputs, clean_labels, optimizer, steps)
        return

    jitter = {"high": 0.4, "medium": 0.2, "low": 0.1}[strength]

    noise = torch.randn_like(clean_inputs) * jitter * 0.1
    augmented = (clean_inputs + noise).clamp(0.0, 1.0)

    gray_mask = torch.rand(len(clean_inputs)) < 0.5
    gray = clean_inputs.mean(dim=1, keepdim=True).expand_as(clean_inputs)
    augmented[gray_mask] = gray[gray_mask]

    mixed = torch.cat([clean_inputs, augmented], dim=0)
    mixed_labels = list(clean_labels) + list(clean_labels)
    _fine_tune_step(model, mixed, mixed_labels, optimizer, steps)


def counterfactual_generation(
    model: BaseModel,
    cluster: VulnerabilityCluster,
    clean_inputs: torch.Tensor,
    clean_labels: List[int],
    strength: str,
    steps: int,
):
    """Fine-tune on background-masked inputs to break background dependency."""
    optimizer = _get_optimizer(model, strength)

    if "distilbert" in model.model_name:
        _fine_tune_step(model, clean_inputs, clean_labels, optimizer, steps)
        return

    mask_fraction = {"high": 0.5, "medium": 0.35, "low": 0.2}[strength]
    h, w = clean_inputs.shape[-2], clean_inputs.shape[-1]
    border = max(1, int(min(h, w) * mask_fraction / 2))

    counterfactuals = clean_inputs.clone()
    counterfactuals[:, :, :border, :] = torch.rand_like(counterfactuals[:, :, :border, :])
    counterfactuals[:, :, -border:, :] = torch.rand_like(counterfactuals[:, :, -border:, :])
    counterfactuals[:, :, :, :border] = torch.rand_like(counterfactuals[:, :, :, :border])
    counterfactuals[:, :, :, -border:] = torch.rand_like(counterfactuals[:, :, :, -border:])

    mixed = torch.cat([clean_inputs, counterfactuals], dim=0)
    mixed_labels = list(clean_labels) + list(clean_labels)
    _fine_tune_step(model, mixed, mixed_labels, optimizer, steps)


def targeted_augmentation(
    model: BaseModel,
    cluster: VulnerabilityCluster,
    clean_inputs: torch.Tensor,
    clean_labels: List[int],
    strength: str,
    steps: int,
):
    """Oversample cluster's original inputs with augmentation to cover edge cases."""
    optimizer = _get_optimizer(model, strength)

    if "distilbert" in model.model_name:
        _fine_tune_step(model, clean_inputs, clean_labels, optimizer, steps)
        return

    if cluster.examples:
        n = min(len(cluster.examples), len(clean_inputs))
        edge_inputs = torch.stack([e.original_input for e in cluster.examples[:n]])
        edge_labels = [e.true_label for e in cluster.examples[:n]]
    else:
        edge_inputs = clean_inputs
        edge_labels = list(clean_labels)

    # Pad/trim to float if needed and match channel/spatial dims of clean_inputs
    if edge_inputs.shape[1:] != clean_inputs.shape[1:]:
        edge_inputs = clean_inputs
        edge_labels = list(clean_labels)

    jitter = torch.randn_like(edge_inputs.float()) * 0.05
    augmented = (edge_inputs.float() + jitter).clamp(0.0, 1.0)

    mixed = torch.cat([clean_inputs, augmented], dim=0)
    mixed_labels = list(clean_labels) + list(edge_labels)
    _fine_tune_step(model, mixed, mixed_labels, optimizer, steps)


STRATEGY_FN = {
    "adversarial_training": adversarial_training,
    "stylized_augmentation": stylized_augmentation,
    "counterfactual_generation": counterfactual_generation,
    "targeted_augmentation": targeted_augmentation,
}
