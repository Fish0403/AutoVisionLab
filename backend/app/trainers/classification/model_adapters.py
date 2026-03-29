"""Model-specific adapter hooks for the generic classification trainer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from app.schemas.parameter_space import ExperimentConfig
from app.trainers.builder import build_model_from_config


CreateModelFn = Callable[[ExperimentConfig], nn.Module]
ComputeLossFn = Callable[[ExperimentConfig, object, torch.Tensor, nn.Module, Callable[[object], torch.Tensor]], torch.Tensor]


def _build_model(config: ExperimentConfig) -> nn.Module:
    """Build one model from the unified recipe builder."""
    return build_model_from_config(config)


def _compute_default_loss(
    config: ExperimentConfig,
    outputs: object,
    labels: torch.Tensor,
    criterion: nn.Module,
    unwrap_logits: Callable[[object], torch.Tensor],
) -> torch.Tensor:
    """Compute the default classification loss from normalized logits."""
    del config
    return criterion(unwrap_logits(outputs), labels)


def _compute_googlenet_loss(
    config: ExperimentConfig,
    outputs: object,
    labels: torch.Tensor,
    criterion: nn.Module,
    unwrap_logits: Callable[[object], torch.Tensor],
) -> torch.Tensor:
    """Include auxiliary head losses when GoogLeNet aux logits are enabled."""
    if config.use_aux_logits() and hasattr(outputs, "aux_logits1") and hasattr(outputs, "aux_logits2"):
        main_loss = criterion(unwrap_logits(outputs), labels)
        aux1_loss = criterion(outputs.aux_logits1, labels)
        aux2_loss = criterion(outputs.aux_logits2, labels)
        return main_loss + 0.3 * (aux1_loss + aux2_loss)
    return _compute_default_loss(config, outputs, labels, criterion, unwrap_logits)


@dataclass(frozen=True)
class ClassificationModelAdapter:
    """Model-specific hooks used by the generic classification trainer."""

    name: str
    create_model: CreateModelFn = _build_model
    compute_loss: ComputeLossFn = _compute_default_loss


CLASSIFICATION_MODEL_ADAPTERS: dict[str, ClassificationModelAdapter] = {
    "mobilenet_v3_small": ClassificationModelAdapter(name="mobilenet_v3_small"),
    "googlenet": ClassificationModelAdapter(name="googlenet", compute_loss=_compute_googlenet_loss),
    "resnet18": ClassificationModelAdapter(name="resnet18"),
}


def get_classification_model_adapter(model_name: str) -> ClassificationModelAdapter:
    """Return the adapter registered for one supported classification model."""
    try:
        return CLASSIFICATION_MODEL_ADAPTERS[model_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model_name: {model_name}") from exc
