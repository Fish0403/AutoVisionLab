"""Backbone builders and default backbone layer payloads."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Any

from torch import nn
from torchvision.models.mobilenetv3 import Conv2dNormActivation, InvertedResidual, InvertedResidualConfig


MOBILENET_V3_SMALL_NATIVE_BACKBONE_LAYERS: tuple[dict[str, Any], ...] = (
    {"from": -1, "repeat": 1, "module": "stem_conv", "args": [16, 3, 2, "hardswish"], "tag": "stem"},
    {"from": -1, "repeat": 1, "module": "inverted_residual", "args": [16, 3, 16, 16, True, "RE", 2, 1], "tag": "C1"},
    {"from": -1, "repeat": 1, "module": "inverted_residual", "args": [16, 3, 72, 24, False, "RE", 2, 1], "tag": "C2"},
    {"from": -1, "repeat": 1, "module": "inverted_residual", "args": [24, 3, 88, 24, False, "RE", 1, 1], "tag": "C2"},
    {"from": -1, "repeat": 1, "module": "inverted_residual", "args": [24, 5, 96, 40, True, "HS", 2, 1], "tag": "C3"},
    {"from": -1, "repeat": 1, "module": "inverted_residual", "args": [40, 5, 240, 40, True, "HS", 1, 1], "tag": "C3"},
    {"from": -1, "repeat": 1, "module": "inverted_residual", "args": [40, 5, 240, 40, True, "HS", 1, 1], "tag": "C3"},
    {"from": -1, "repeat": 1, "module": "inverted_residual", "args": [40, 5, 120, 48, True, "HS", 1, 1], "tag": "C3"},
    {"from": -1, "repeat": 1, "module": "inverted_residual", "args": [48, 5, 144, 48, True, "HS", 1, 1], "tag": "C3"},
    {"from": -1, "repeat": 1, "module": "inverted_residual", "args": [48, 5, 288, 96, True, "HS", 2, 1], "tag": "C4"},
    {"from": -1, "repeat": 1, "module": "inverted_residual", "args": [96, 5, 576, 96, True, "HS", 1, 1], "tag": "C4"},
    {"from": -1, "repeat": 1, "module": "inverted_residual", "args": [96, 5, 576, 96, True, "HS", 1, 1], "tag": "C4"},
)


@dataclass(frozen=True)
class MobileNetV3BackboneBuildResult:
    """Intermediate feature stack built from the MobileNet backbone recipe."""

    features: list[nn.Module]
    norm_layer: type[nn.Module]
    output_channels: int


def resolve_mobilenet_activation_layer(activation_type: str) -> type[nn.Module]:
    """Return one activation layer class from the MobileNet recipe string."""
    if activation_type == "hardswish":
        return nn.Hardswish
    if activation_type == "relu":
        return nn.ReLU
    raise ValueError(f"Unsupported activation_type for MobileNetV3 Small: {activation_type}")


def build_default_backbone_layers(base_model: str) -> list[dict[str, Any]]:
    """Return default backbone layer payloads for one supported base model."""
    if base_model == "mobilenet_v3_small":
        return deepcopy(list(MOBILENET_V3_SMALL_NATIVE_BACKBONE_LAYERS))
    return []


def build_mobilenet_v3_small_native_backbone(recipe: Any) -> MobileNetV3BackboneBuildResult:
    """Build the MobileNetV3 Small native backbone feature extractor."""
    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)
    stem_layer = recipe.backbone[0]
    stem_out_channels, stem_kernel_size, stem_stride, stem_activation_type = stem_layer.args
    stem_activation = resolve_mobilenet_activation_layer(str(stem_activation_type))
    stem_output_channels = InvertedResidualConfig.adjust_channels(
        int(stem_out_channels),
        recipe.width_multiple,
    )
    features: list[nn.Module] = [
        Conv2dNormActivation(
            recipe.input_channels,
            stem_output_channels,
            kernel_size=int(stem_kernel_size),
            stride=int(stem_stride),
            norm_layer=norm_layer,
            activation_layer=stem_activation,
        )
    ]

    for layer in recipe.backbone[1:]:
        input_channels, kernel_size, expanded_channels, out_channels, use_se, activation, stride, dilation = layer.args
        features.append(
            InvertedResidual(
                InvertedResidualConfig(
                    input_channels,
                    kernel_size,
                    expanded_channels,
                    out_channels,
                    bool(use_se),
                    str(activation),
                    stride,
                    dilation,
                    recipe.width_multiple,
                ),
                norm_layer,
            )
        )

    output_channels = InvertedResidualConfig.adjust_channels(
        int(recipe.backbone[-1].args[3]),
        recipe.width_multiple,
    )
    return MobileNetV3BackboneBuildResult(
        features=features,
        norm_layer=norm_layer,
        output_channels=output_channels,
    )
