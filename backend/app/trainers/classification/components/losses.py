"""Loss registry for structured classification training."""

from __future__ import annotations

from typing import TypeAlias

import torch
import torch.nn.functional as functional
from torch import nn

from app.schemas.parameter_space import ExperimentParams


MixedTargets: TypeAlias = tuple[torch.Tensor, torch.Tensor, float]


class ClassificationLoss(nn.Module):
    """Base loss wrapper that supports mixed-label augmentations."""

    def forward(self, logits: torch.Tensor, targets: torch.Tensor | MixedTargets) -> torch.Tensor:
        if isinstance(targets, tuple):
            primary_targets, secondary_targets, mix_ratio = targets
            return (
                mix_ratio * self.forward(logits, primary_targets)
                + (1.0 - mix_ratio) * self.forward(logits, secondary_targets)
            )
        return self.compute_loss(logits, targets)

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute one concrete loss value."""
        raise NotImplementedError


class CrossEntropyClassificationLoss(ClassificationLoss):
    """Cross-entropy loss with optional label smoothing."""

    def __init__(self, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return functional.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)


class FocalLoss(ClassificationLoss):
    """Multi-class focal loss on top of cross entropy."""

    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        per_sample_loss = functional.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        probabilities = torch.exp(-per_sample_loss)
        focal_weight = (1.0 - probabilities).pow(self.gamma)
        return (focal_weight * per_sample_loss).mean()


def build_loss(params: ExperimentParams) -> ClassificationLoss:
    """Build one structured loss implementation."""
    if params.loss_name == "focal_loss":
        return FocalLoss(gamma=params.loss_params.focal_gamma, label_smoothing=params.label_smoothing)
    if params.loss_name == "cross_entropy":
        return CrossEntropyClassificationLoss(label_smoothing=0.0)
    return CrossEntropyClassificationLoss(label_smoothing=params.label_smoothing)
