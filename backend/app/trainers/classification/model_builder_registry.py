"""Generic classification model builder registry and native adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torchvision.models import (
    efficientnet_b0,
    efficientnet_b1,
    googlenet,
    mobilenet_v2,
    mobilenet_v3_large,
    mobilenet_v3_small,
    resnet18,
    resnet34,
    resnet50,
)

from app.schemas.parameter_space import ModelRecipe, hydrate_model_recipe
from app.trainers.classification.model_components import build_classification_neck


ValidateRecipeFn = Callable[[ModelRecipe], None]
BuildModelFn = Callable[[ModelRecipe, int], nn.Module]


class GenericTorchvisionClassifier(nn.Module):
    """Assemble one standard torchvision classifier from backbone, neck, and head."""

    def __init__(self, *, backbone: nn.Module, neck: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run one forward pass for a standard classification model."""
        outputs = self.backbone(inputs)
        outputs = self.neck(outputs)
        outputs = torch.flatten(outputs, 1)
        return self.head(outputs)


@dataclass(frozen=True)
class ClassificationModelBuilderAdapter:
    """Recipe validation and model construction hooks for one base model."""

    name: str
    validate_recipe: ValidateRecipeFn
    build_model: BuildModelFn


@dataclass(frozen=True)
class TorchvisionClassifierAdapter:
    """Describe one standard torchvision classifier that can use the generic builder."""

    name: str
    backbone_component_name: str
    build_native_model: Callable[[int, float], nn.Module]
    build_backbone: Callable[[nn.Module], nn.Module]
    build_native_head: Callable[[nn.Module], nn.Module]
    feature_dim: int
    default_dropout: float = 0.0


def _build_native_linear_head(*, feature_dim: int, num_classes: int) -> nn.Module:
    """Build one standard native linear classifier head."""
    return nn.Linear(feature_dim, num_classes)


def _build_native_dropout_linear_head(
    *,
    feature_dim: int,
    num_classes: int,
    dropout_probability: float,
) -> nn.Module:
    """Build one standard dropout-linear classifier head."""
    if dropout_probability <= 0:
        return _build_native_linear_head(feature_dim=feature_dim, num_classes=num_classes)
    return nn.Sequential(
        nn.Dropout(p=dropout_probability, inplace=True),
        nn.Linear(feature_dim, num_classes),
    )


def _build_standard_head(
    *,
    adapter: TorchvisionClassifierAdapter,
    native_model: nn.Module,
    head_name: str,
    feature_dim: int,
    num_classes: int,
    dropout_probability: float,
) -> nn.Module:
    """Build one supported classifier head for the generic torchvision wrapper."""
    if head_name == "native_classifier":
        return adapter.build_native_head(native_model)
    if head_name == "linear":
        return _build_native_linear_head(feature_dim=feature_dim, num_classes=num_classes)
    if head_name == "dropout_linear":
        return _build_native_dropout_linear_head(
            feature_dim=feature_dim,
            num_classes=num_classes,
            dropout_probability=dropout_probability,
        )
    raise ValueError(f"Unsupported head component for {adapter.name}: {head_name}")


def _validate_standard_torchvision_recipe(
    recipe: ModelRecipe,
    *,
    adapter: TorchvisionClassifierAdapter,
) -> None:
    """Validate the minimal supported recipe subset for one standard torchvision classifier."""
    if recipe.base_model != adapter.name:
        raise ValueError(f"Unsupported base_model for {adapter.name} builder: {recipe.base_model}")
    if recipe.task_type != "classification":
        raise ValueError(f"Unsupported task_type for {adapter.name} builder: {recipe.task_type}")
    if recipe.width_multiple != 1.0:
        raise ValueError(f"{adapter.name} v1 builder does not support width_multiple changes")
    if recipe.backbone_config.stem_variant != "standard":
        raise ValueError(f"Unsupported stem_variant for {adapter.name}: {recipe.backbone_config.stem_variant}")
    if recipe.backbone_config.attention_module != "none":
        raise ValueError(f"Unsupported attention_module for {adapter.name}: {recipe.backbone_config.attention_module}")
    if recipe.backbone_config.last_channel_multiplier != 1.0:
        raise ValueError(f"{adapter.name} v1 builder does not support last_channel_multiplier changes")
    if recipe.head_config.pooling_type not in {"avg", "gem"}:
        raise ValueError(f"Unsupported pooling_type for {adapter.name}: {recipe.head_config.pooling_type}")
    if recipe.head_config.classifier_type != "linear":
        raise ValueError(f"Unsupported classifier_type for {adapter.name}: {recipe.head_config.classifier_type}")
    if recipe.components is not None:
        if recipe.components.backbone.name != adapter.backbone_component_name:
            raise ValueError(
                f"Unsupported backbone component for {adapter.name}: {recipe.components.backbone.name}"
            )
        if recipe.components.neck.name not in {"avg_pool", "gem_pool"}:
            raise ValueError(f"Unsupported neck component for {adapter.name}: {recipe.components.neck.name}")
        if recipe.components.head.name not in {"native_classifier", "linear", "dropout_linear"}:
            raise ValueError(f"Unsupported head component for {adapter.name}: {recipe.components.head.name}")
    if recipe.neck:
        raise ValueError(f"{adapter.name} v1 builder does not support neck configuration")
    if recipe.modules:
        raise ValueError(f"Unsupported extra recipe modules for {adapter.name}: {sorted(recipe.modules.keys())}")


def _build_standard_torchvision_model(
    recipe: ModelRecipe,
    num_classes: int,
    *,
    adapter: TorchvisionClassifierAdapter,
) -> nn.Module:
    """Build one standard torchvision classification model via the generic classifier wrapper."""
    _validate_standard_torchvision_recipe(recipe, adapter=adapter)
    output_classes = recipe.nc or num_classes
    native_model = adapter.build_native_model(output_classes, adapter.default_dropout)
    backbone = adapter.build_backbone(native_model)
    neck_name = _resolve_neck_component_name(recipe)
    head_name = _resolve_head_component_name(recipe)
    neck = build_classification_neck(neck_name)
    head = _build_standard_head(
        adapter=adapter,
        native_model=native_model,
        head_name=head_name,
        feature_dim=adapter.feature_dim,
        num_classes=output_classes,
        dropout_probability=recipe.head_config.classifier_dropout or adapter.default_dropout,
    )
    return GenericTorchvisionClassifier(backbone=backbone, neck=neck, head=head)


def _build_standard_torchvision_builder(adapter: TorchvisionClassifierAdapter) -> ClassificationModelBuilderAdapter:
    """Create one registry adapter for a standard torchvision classifier."""
    return ClassificationModelBuilderAdapter(
        name=adapter.name,
        validate_recipe=lambda recipe, adapter=adapter: _validate_standard_torchvision_recipe(
            recipe,
            adapter=adapter,
        ),
        build_model=lambda recipe, num_classes, adapter=adapter: _build_standard_torchvision_model(
            recipe,
            num_classes,
            adapter=adapter,
        ),
    )


def _build_resnet_backbone(native_model: nn.Module) -> nn.Module:
    """Extract the feature backbone from one torchvision ResNet model."""
    return nn.Sequential(
        native_model.conv1,
        native_model.bn1,
        native_model.relu,
        native_model.maxpool,
        native_model.layer1,
        native_model.layer2,
        native_model.layer3,
        native_model.layer4,
    )


def _build_mobilenet_v2_backbone(native_model: nn.Module) -> nn.Module:
    """Extract the feature backbone from one torchvision MobileNetV2 model."""
    return native_model.features


def _build_mobilenet_v3_backbone(native_model: nn.Module) -> nn.Module:
    """Extract the feature backbone from one torchvision MobileNetV3 model."""
    return native_model.features


def _build_efficientnet_backbone(native_model: nn.Module) -> nn.Module:
    """Extract the feature backbone from one torchvision EfficientNet model."""
    return native_model.features


TORCHVISION_CLASSIFIER_ADAPTERS = {
    "mobilenet_v2": TorchvisionClassifierAdapter(
        name="mobilenet_v2",
        backbone_component_name="mobilenet_v2_native",
        build_native_model=lambda num_classes, dropout: mobilenet_v2(
            num_classes=num_classes,
            dropout=dropout,
        ),
        build_backbone=_build_mobilenet_v2_backbone,
        build_native_head=lambda native_model: native_model.classifier,
        feature_dim=1280,
        default_dropout=0.2,
    ),
    "mobilenet_v3_small": TorchvisionClassifierAdapter(
        name="mobilenet_v3_small",
        backbone_component_name="mobilenet_v3_small_native",
        build_native_model=lambda num_classes, dropout: mobilenet_v3_small(
            num_classes=num_classes,
            dropout=dropout,
        ),
        build_backbone=_build_mobilenet_v3_backbone,
        build_native_head=lambda native_model: native_model.classifier,
        feature_dim=576,
        default_dropout=0.2,
    ),
    "mobilenet_v3_large": TorchvisionClassifierAdapter(
        name="mobilenet_v3_large",
        backbone_component_name="mobilenet_v3_large_native",
        build_native_model=lambda num_classes, dropout: mobilenet_v3_large(
            num_classes=num_classes,
            dropout=dropout,
        ),
        build_backbone=_build_mobilenet_v3_backbone,
        build_native_head=lambda native_model: native_model.classifier,
        feature_dim=960,
        default_dropout=0.2,
    ),
    "efficientnet_b0": TorchvisionClassifierAdapter(
        name="efficientnet_b0",
        backbone_component_name="efficientnet_b0_native",
        build_native_model=lambda num_classes, dropout: efficientnet_b0(
            num_classes=num_classes,
            dropout=dropout,
        ),
        build_backbone=_build_efficientnet_backbone,
        build_native_head=lambda native_model: native_model.classifier,
        feature_dim=1280,
        default_dropout=0.2,
    ),
    "efficientnet_b1": TorchvisionClassifierAdapter(
        name="efficientnet_b1",
        backbone_component_name="efficientnet_b1_native",
        build_native_model=lambda num_classes, dropout: efficientnet_b1(
            num_classes=num_classes,
            dropout=dropout,
        ),
        build_backbone=_build_efficientnet_backbone,
        build_native_head=lambda native_model: native_model.classifier,
        feature_dim=1280,
        default_dropout=0.2,
    ),
    "resnet18": TorchvisionClassifierAdapter(
        name="resnet18",
        backbone_component_name="resnet18_native",
        build_native_model=lambda num_classes, _dropout: resnet18(num_classes=num_classes),
        build_backbone=_build_resnet_backbone,
        build_native_head=lambda native_model: native_model.fc,
        feature_dim=512,
        default_dropout=0.0,
    ),
    "resnet34": TorchvisionClassifierAdapter(
        name="resnet34",
        backbone_component_name="resnet34_native",
        build_native_model=lambda num_classes, _dropout: resnet34(num_classes=num_classes),
        build_backbone=_build_resnet_backbone,
        build_native_head=lambda native_model: native_model.fc,
        feature_dim=512,
        default_dropout=0.0,
    ),
    "resnet50": TorchvisionClassifierAdapter(
        name="resnet50",
        backbone_component_name="resnet50_native",
        build_native_model=lambda num_classes, _dropout: resnet50(num_classes=num_classes),
        build_backbone=_build_resnet_backbone,
        build_native_head=lambda native_model: native_model.fc,
        feature_dim=2048,
        default_dropout=0.0,
    ),
}


def _resolve_head_component_name(recipe: ModelRecipe) -> str:
    """Return the active head component name, defaulting to the native classifier."""
    if recipe.components is None:
        return "native_classifier"
    return recipe.components.head.name


def _resolve_neck_component_name(recipe: ModelRecipe) -> str:
    """Return the active neck component name, defaulting to average pooling."""
    if recipe.components is None:
        return "avg_pool"
    return recipe.components.neck.name


def _validate_googlenet_recipe(recipe: ModelRecipe) -> None:
    """Validate the minimal supported GoogLeNet recipe subset."""
    if recipe.base_model != "googlenet":
        raise ValueError(f"Unsupported base_model for GoogLeNet builder: {recipe.base_model}")
    if recipe.task_type != "classification":
        raise ValueError(f"Unsupported task_type for GoogLeNet builder: {recipe.task_type}")
    if recipe.width_multiple != 1.0:
        raise ValueError("GoogLeNet v1 builder does not support width_multiple changes")
    if recipe.backbone_config.stem_variant != "standard":
        raise ValueError(f"Unsupported stem_variant for GoogLeNet: {recipe.backbone_config.stem_variant}")
    if recipe.backbone_config.attention_module != "none":
        raise ValueError(f"Unsupported attention_module for GoogLeNet: {recipe.backbone_config.attention_module}")
    if recipe.backbone_config.last_channel_multiplier != 1.0:
        raise ValueError("GoogLeNet v1 builder does not support last_channel_multiplier changes")
    if recipe.head_config.pooling_type != "avg":
        raise ValueError(f"Unsupported pooling_type for GoogLeNet: {recipe.head_config.pooling_type}")
    if recipe.head_config.classifier_type != "linear":
        raise ValueError(f"Unsupported classifier_type for GoogLeNet: {recipe.head_config.classifier_type}")
    if recipe.components is not None:
        if recipe.components.backbone.name != "googlenet_native":
            raise ValueError(f"Unsupported backbone component for GoogLeNet: {recipe.components.backbone.name}")
        if recipe.components.neck.name != "avg_pool":
            raise ValueError(f"Unsupported neck component for GoogLeNet: {recipe.components.neck.name}")
        if recipe.components.head.name != "native_classifier":
            raise ValueError(f"Unsupported head component for GoogLeNet: {recipe.components.head.name}")
    if recipe.neck:
        raise ValueError("GoogLeNet v1 builder does not support neck configuration")
    if recipe.backbone or recipe.head:
        raise ValueError("GoogLeNet v1 builder does not support custom architecture layers")
    extra_modules = sorted(set(recipe.modules.keys()) - {"aux_logits"})
    if extra_modules:
        raise ValueError(f"Unsupported extra recipe modules for GoogLeNet: {extra_modules}")


def _build_googlenet_from_recipe(recipe: ModelRecipe, num_classes: int) -> nn.Module:
    """Build one GoogLeNet model from the supported recipe subset."""
    _validate_googlenet_recipe(recipe)
    return googlenet(
        num_classes=recipe.nc or num_classes,
        aux_logits=bool(recipe.modules.get("aux_logits", False)),
        dropout=recipe.head_config.classifier_dropout,
    )


CLASSIFICATION_MODEL_BUILDERS = {
    "googlenet": ClassificationModelBuilderAdapter(
        name="googlenet",
        validate_recipe=_validate_googlenet_recipe,
        build_model=_build_googlenet_from_recipe,
    ),
}
CLASSIFICATION_MODEL_BUILDERS.update(
    {
        model_name: _build_standard_torchvision_builder(adapter)
        for model_name, adapter in TORCHVISION_CLASSIFIER_ADAPTERS.items()
    }
)


def get_classification_model_builder(recipe: ModelRecipe) -> ClassificationModelBuilderAdapter:
    """Return the builder adapter registered for one model recipe."""
    try:
        return CLASSIFICATION_MODEL_BUILDERS[recipe.base_model]
    except KeyError as exc:
        raise ValueError(f"Unsupported model recipe base_model: {recipe.base_model}") from exc


def validate_classification_model_recipe(recipe: ModelRecipe) -> None:
    """Validate one structured classification model recipe."""
    hydrated_recipe = hydrate_model_recipe(recipe)
    get_classification_model_builder(hydrated_recipe).validate_recipe(hydrated_recipe)


def build_classification_model_from_recipe(recipe: ModelRecipe, *, num_classes: int = 10) -> nn.Module:
    """Build one classification model from one validated model recipe."""
    hydrated_recipe = hydrate_model_recipe(recipe)
    validate_classification_model_recipe(hydrated_recipe)
    return get_classification_model_builder(hydrated_recipe).build_model(hydrated_recipe, num_classes)
