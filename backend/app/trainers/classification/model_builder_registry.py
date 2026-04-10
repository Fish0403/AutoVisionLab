"""Generic classification model builder registry and native adapters."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import torch
import torchvision.models as torchvision_models
from torch import nn

from app.model_catalog.registry import get_model_catalog_entry
from app.model_catalog.schemas import ModelManifest, TorchvisionClassifierBuilderSpec
from app.schemas.parameter_space import ModelRecipe, hydrate_model_recipe
from app.trainers.classification.model_components.necks import build_classification_neck


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


def _resolve_attr_path(target: object, attr_path: str) -> object:
    """Resolve one dotted attribute path on one object."""
    current = target
    for attr_name in attr_path.split("."):
        current = getattr(current, attr_name)
    return current


def _build_standard_head(
    *,
    native_model: nn.Module,
    native_head_attr: str,
    head_name: str,
    feature_dim: int,
    num_classes: int,
    dropout_probability: float,
) -> nn.Module:
    """Build one supported classifier head for the generic torchvision wrapper."""
    if head_name == "native_classifier":
        native_head = _resolve_attr_path(native_model, native_head_attr)
        if not isinstance(native_head, nn.Module):
            raise ValueError(f"Resolved native head is not an nn.Module: {native_head_attr}")
        return native_head
    if head_name == "linear":
        return _build_native_linear_head(feature_dim=feature_dim, num_classes=num_classes)
    if head_name == "dropout_linear":
        return _build_native_dropout_linear_head(
            feature_dim=feature_dim,
            num_classes=num_classes,
            dropout_probability=dropout_probability,
        )
    raise ValueError(f"Unsupported head component: {head_name}")


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


def _build_backbone_from_spec(native_model: nn.Module, spec: TorchvisionClassifierBuilderSpec) -> nn.Module:
    """Extract one feature backbone using the declared manifest profile."""
    if spec.backbone_extractor == "feature_sequence":
        return getattr(native_model, "features")
    if spec.backbone_extractor == "feature_sequence_with_avgpool":
        return nn.Sequential(getattr(native_model, "features"), getattr(native_model, "avgpool"))
    if spec.backbone_extractor == "resnet_stages":
        return _build_resnet_backbone(native_model)
    raise ValueError(f"Unsupported backbone_extractor: {spec.backbone_extractor}")


def _build_torchvision_native_model(
    spec: TorchvisionClassifierBuilderSpec,
    *,
    num_classes: int,
) -> nn.Module:
    """Build one torchvision native model from the declared manifest spec."""
    try:
        model_factory = getattr(torchvision_models, spec.torchvision_name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported torchvision model: {spec.torchvision_name}") from exc

    kwargs: dict[str, object] = {"num_classes": num_classes}
    if spec.uses_dropout_arg:
        kwargs["dropout"] = spec.default_dropout
    return model_factory(**kwargs)


def _resolve_head_component_name(recipe: ModelRecipe, *, default_head_name: str = "native_classifier") -> str:
    """Return the active head component name."""
    if recipe.components is None:
        return default_head_name
    return recipe.components.head.name


def _resolve_neck_component_name(recipe: ModelRecipe, *, default_neck_name: str = "avg_pool") -> str:
    """Return the active neck component name."""
    if recipe.components is None:
        return default_neck_name
    return recipe.components.neck.name


def _validate_torchvision_recipe(recipe: ModelRecipe, *, spec: TorchvisionClassifierBuilderSpec) -> None:
    """Validate the supported recipe subset for one manifest-declared torchvision model."""
    if recipe.task_type != "classification":
        raise ValueError(f"Unsupported task_type for {recipe.base_model} builder: {recipe.task_type}")
    if recipe.width_multiple != 1.0:
        raise ValueError(f"{recipe.base_model} v1 builder does not support width_multiple changes")
    if recipe.backbone_config.stem_variant != "standard":
        raise ValueError(f"Unsupported stem_variant for {recipe.base_model}: {recipe.backbone_config.stem_variant}")
    if recipe.backbone_config.attention_module != "none":
        raise ValueError(
            f"Unsupported attention_module for {recipe.base_model}: {recipe.backbone_config.attention_module}"
        )
    if recipe.backbone_config.last_channel_multiplier != 1.0:
        raise ValueError(f"{recipe.base_model} v1 builder does not support last_channel_multiplier changes")
    if recipe.head_config.pooling_type not in {"avg", "gem"}:
        raise ValueError(f"Unsupported pooling_type for {recipe.base_model}: {recipe.head_config.pooling_type}")
    if recipe.head_config.classifier_type != "linear":
        raise ValueError(f"Unsupported classifier_type for {recipe.base_model}: {recipe.head_config.classifier_type}")
    if recipe.components is not None:
        if recipe.components.backbone.name != spec.backbone_component_name:
            raise ValueError(
                f"Unsupported backbone component for {recipe.base_model}: {recipe.components.backbone.name}"
            )
        if recipe.components.neck.name not in set(spec.supported_neck_names):
            raise ValueError(f"Unsupported neck component for {recipe.base_model}: {recipe.components.neck.name}")
        if recipe.components.head.name not in set(spec.supported_head_names):
            raise ValueError(f"Unsupported head component for {recipe.base_model}: {recipe.components.head.name}")
    if recipe.neck:
        raise ValueError(f"{recipe.base_model} v1 builder does not support neck configuration")
    if recipe.backbone or recipe.head:
        raise ValueError(f"{recipe.base_model} v1 builder does not support custom architecture layers")
    if recipe.modules:
        raise ValueError(f"Unsupported extra recipe modules for {recipe.base_model}: {sorted(recipe.modules.keys())}")


def _build_torchvision_model(
    recipe: ModelRecipe,
    num_classes: int,
    *,
    spec: TorchvisionClassifierBuilderSpec,
) -> nn.Module:
    """Build one standard torchvision classification model via the generic wrapper."""
    _validate_torchvision_recipe(recipe, spec=spec)
    output_classes = recipe.nc or num_classes
    native_model = _build_torchvision_native_model(spec, num_classes=output_classes)
    backbone = _build_backbone_from_spec(native_model, spec)
    neck_name = _resolve_neck_component_name(recipe, default_neck_name=spec.default_neck_name)
    head_name = _resolve_head_component_name(recipe, default_head_name=spec.default_head_name)
    neck = build_classification_neck(neck_name)
    head = _build_standard_head(
        native_model=native_model,
        native_head_attr=spec.native_head_attr,
        head_name=head_name,
        feature_dim=spec.feature_dim,
        num_classes=output_classes,
        dropout_probability=recipe.head_config.classifier_dropout or spec.default_dropout,
    )
    return GenericTorchvisionClassifier(backbone=backbone, neck=neck, head=head)


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
    return torchvision_models.googlenet(
        num_classes=recipe.nc or num_classes,
        aux_logits=bool(recipe.modules.get("aux_logits", False)),
        dropout=recipe.head_config.classifier_dropout,
    )


@lru_cache(maxsize=None)
def _build_torchvision_builder(model_name: str) -> ClassificationModelBuilderAdapter:
    """Build one runtime builder adapter from the catalog entry."""
    model_entry = get_model_catalog_entry(model_name)
    if model_entry is None or model_entry.task_type != "classification":
        raise ValueError(f"Unsupported model recipe base_model: {model_name}")
    if model_entry.builder.type != "torchvision_classifier":
        raise ValueError(f"Model {model_name} is not declared as a torchvision classifier")
    spec = model_entry.builder
    return ClassificationModelBuilderAdapter(
        name=model_name,
        validate_recipe=lambda recipe, spec=spec: _validate_torchvision_recipe(recipe, spec=spec),
        build_model=lambda recipe, num_classes, spec=spec: _build_torchvision_model(
            recipe,
            num_classes,
            spec=spec,
        ),
    )


def get_classification_model_builder(recipe: ModelRecipe) -> ClassificationModelBuilderAdapter:
    """Return the builder adapter registered for one model recipe."""
    model_entry = get_model_catalog_entry(recipe.base_model)
    if model_entry is None or model_entry.task_type != "classification":
        raise ValueError(f"Unsupported model recipe base_model: {recipe.base_model}")
    if model_entry.builder.type == "googlenet_classifier":
        return ClassificationModelBuilderAdapter(
            name=recipe.base_model,
            validate_recipe=_validate_googlenet_recipe,
            build_model=_build_googlenet_from_recipe,
        )
    if model_entry.builder.type == "torchvision_classifier":
        return _build_torchvision_builder(recipe.base_model)
    raise ValueError(f"Unsupported builder type for model recipe base_model: {recipe.base_model}")


def validate_classification_model_recipe(recipe: ModelRecipe) -> None:
    """Validate one structured classification model recipe."""
    hydrated_recipe = hydrate_model_recipe(recipe)
    get_classification_model_builder(hydrated_recipe).validate_recipe(hydrated_recipe)


def build_classification_model_from_recipe(recipe: ModelRecipe, *, num_classes: int = 10) -> nn.Module:
    """Build one classification model from one validated model recipe."""
    hydrated_recipe = hydrate_model_recipe(recipe)
    validate_classification_model_recipe(hydrated_recipe)
    return get_classification_model_builder(hydrated_recipe).build_model(hydrated_recipe, num_classes)


def validate_classification_model_manifest(manifest: ModelManifest) -> None:
    """Validate one not-yet-registered manifest against the current builder rules."""
    hydrated_recipe = hydrate_model_recipe(manifest.default_model_recipe)
    if manifest.builder.type == "googlenet_classifier":
        _validate_googlenet_recipe(hydrated_recipe)
        return
    if manifest.builder.type == "torchvision_classifier":
        _validate_torchvision_recipe(hydrated_recipe, spec=manifest.builder)
        return
    raise ValueError(f"Unsupported builder type for model manifest: {manifest.builder.type}")


def build_classification_model_from_manifest(manifest: ModelManifest, *, num_classes: int = 10) -> nn.Module:
    """Build one classification model directly from one manifest draft."""
    hydrated_recipe = hydrate_model_recipe(manifest.default_model_recipe)
    validate_classification_model_manifest(manifest)
    if manifest.builder.type == "googlenet_classifier":
        return _build_googlenet_from_recipe(hydrated_recipe, num_classes)
    if manifest.builder.type == "torchvision_classifier":
        return _build_torchvision_model(hydrated_recipe, num_classes, spec=manifest.builder)
    raise ValueError(f"Unsupported builder type for model manifest: {manifest.builder.type}")
