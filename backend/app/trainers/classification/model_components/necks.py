"""Neck modules used by classification model builders."""

from __future__ import annotations

import torch
from torch import nn


class GeMPooling2d(nn.Module):
    """Generalized mean pooling used as one lightweight neck variant."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = p
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Pool one feature map tensor into a global descriptor."""
        inputs = inputs.clamp(min=self.eps).pow(self.p)
        pooled = inputs.mean(dim=(-2, -1), keepdim=True)
        return pooled.pow(1.0 / self.p)


def build_classification_neck(neck_name: str) -> nn.Module:
    """Build one supported classification neck module."""
    if neck_name == "avg_pool":
        return nn.AdaptiveAvgPool2d(1)
    if neck_name == "gem_pool":
        return GeMPooling2d()
    raise ValueError(f"Unsupported classification neck component: {neck_name}")
