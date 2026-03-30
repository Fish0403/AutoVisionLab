"""Head builders and default head layer payloads."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from torch import nn
from torchvision.models.mobilenetv3 import Conv2dNormActivation, InvertedResidualConfig

from app.trainers.classification.model_components.backbones import resolve_mobilenet_activation_layer


MOBILENET_V3_SMALL_DEFAULT_HEAD_LAYERS: tuple[dict[str, Any], ...] = (
    {"from": -1, "repeat": 1, "module": "pointwise_tail", "args": [6, "hardswish"], "tag": "tail"},
    {"from": -1, "repeat": 1, "module": "global_pool", "args": [], "tag": "pool"},
    {"from": -1, "repeat": 1, "module": "classifier", "args": [1024], "tag": "cls"},
)


def build_default_head_layers(base_model: str) -> list[dict[str, Any]]:
    """Return default head layer payloads for one supported base model."""
    if base_model == "mobilenet_v3_small":
        return deepcopy(list(MOBILENET_V3_SMALL_DEFAULT_HEAD_LAYERS))
    return []


def _find_recipe_layer(head_layers: list[object], module_name: str) -> list[object]:
    """Return the args for one named architecture layer."""
    for layer in head_layers:
        if layer.module == module_name:
            return layer.args
    raise ValueError(f"MobileNetV3 Small recipe head is missing {module_name}")


def build_mobilenet_v3_small_tail(
    recipe: Any,
    *,
    input_channels: int,
    norm_layer: type[nn.Module],
) -> tuple[nn.Module, int]:
    """Build the MobileNetV3 tail projection before neck pooling."""
    tail_args = _find_recipe_layer(recipe.head, "pointwise_tail")
    expansion_factor = int(tail_args[0])
    tail_activation = resolve_mobilenet_activation_layer(str(tail_args[1]))
    output_channels = expansion_factor * input_channels
    return (
        Conv2dNormActivation(
            input_channels,
            output_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=tail_activation,
        ),
        output_channels,
    )


def _build_native_classifier(
    *,
    lastconv_output_channels: int,
    classifier_hidden_channels: int,
    classifier_dropout: float,
    num_classes: int,
) -> nn.Sequential:
    """Build the native MobileNetV3 classifier head."""
    return nn.Sequential(
        nn.Linear(lastconv_output_channels, classifier_hidden_channels),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=classifier_dropout, inplace=True),
        nn.Linear(classifier_hidden_channels, num_classes),
    )


def _build_linear_classifier(
    *,
    lastconv_output_channels: int,
    num_classes: int,
) -> nn.Linear:
    """Build one pooled linear classifier head."""
    return nn.Linear(lastconv_output_channels, num_classes)


def _build_dropout_linear_classifier(
    *,
    lastconv_output_channels: int,
    num_classes: int,
    dropout_probability: float,
) -> nn.Sequential:
    """Build one pooled dropout-linear classifier head."""
    return nn.Sequential(
        nn.Dropout(p=dropout_probability, inplace=True),
        nn.Linear(lastconv_output_channels, num_classes),
    )


def build_mobilenet_v3_small_classifier(
    recipe: Any,
    *,
    head_name: str,
    lastconv_output_channels: int,
    num_classes: int,
) -> nn.Module:
    """Build one MobileNetV3 classifier head from the component name."""
    classifier_args = _find_recipe_layer(recipe.head, "classifier")
    classifier_hidden_channels = InvertedResidualConfig.adjust_channels(
        int(classifier_args[0]),
        recipe.width_multiple,
    )
    if head_name == "native_classifier":
        return _build_native_classifier(
            lastconv_output_channels=lastconv_output_channels,
            classifier_hidden_channels=classifier_hidden_channels,
            classifier_dropout=recipe.head_config.classifier_dropout,
            num_classes=num_classes,
        )
    if head_name == "linear":
        return _build_linear_classifier(
            lastconv_output_channels=lastconv_output_channels,
            num_classes=num_classes,
        )
    if head_name == "dropout_linear":
        return _build_dropout_linear_classifier(
            lastconv_output_channels=lastconv_output_channels,
            num_classes=num_classes,
            dropout_probability=0.2,
        )
    raise ValueError(f"Unsupported head component for MobileNetV3 Small: {head_name}")
