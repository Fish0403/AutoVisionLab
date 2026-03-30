"""Generic classification model builder registry and native adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torchvision.models import googlenet, resnet18

from app.schemas.parameter_space import ModelRecipe, hydrate_model_recipe
from app.trainers.classification.model_components import (
    build_classification_neck,
    build_mobilenet_v3_small_classifier,
    build_mobilenet_v3_small_native_backbone,
    build_mobilenet_v3_small_tail,
)


ValidateRecipeFn = Callable[[ModelRecipe], None]
BuildModelFn = Callable[[ModelRecipe, int], nn.Module]


class RecipeMobileNetV3(nn.Module):
    """MobileNetV3 classifier assembled from one structured recipe."""

    def __init__(
        self,
        *,
        features: list[nn.Module],
        avgpool: nn.Module,
        classifier: nn.Module,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(*features)
        self.avgpool = avgpool
        self.classifier = classifier
        self.num_classes = num_classes
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Mirror torchvision MobileNetV3 parameter initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run one forward pass."""
        outputs = self.features(inputs)
        outputs = self.avgpool(outputs)
        outputs = torch.flatten(outputs, 1)
        return self.classifier(outputs)


@dataclass(frozen=True)
class ClassificationModelBuilderAdapter:
    """Recipe validation and model construction hooks for one base model."""

    name: str
    validate_recipe: ValidateRecipeFn
    build_model: BuildModelFn


def _find_recipe_layer(head_layers: list[object], module_name: str) -> list[object]:
    """Return the args for one named architecture layer."""
    for layer in head_layers:
        if layer.module == module_name:
            return layer.args
    raise ValueError(f"MobileNetV3 Small recipe head is missing {module_name}")


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


def _validate_mobilenet_v3_small_recipe(recipe: ModelRecipe) -> None:
    """Validate the supported MobileNetV3 Small recipe subset."""
    if recipe.base_model != "mobilenet_v3_small":
        raise ValueError(f"Unsupported base_model for MobileNet builder: {recipe.base_model}")
    if recipe.task_type != "classification":
        raise ValueError(f"Unsupported task_type for MobileNet builder: {recipe.task_type}")
    if not recipe.backbone:
        raise ValueError("MobileNetV3 Small recipe must define backbone layers")
    if not recipe.head:
        raise ValueError("MobileNetV3 Small recipe must define head layers")
    if recipe.neck:
        raise ValueError("MobileNetV3 Small classification recipe does not support neck modules")
    if recipe.backbone_config.stem_variant != "standard":
        raise ValueError(f"Unsupported stem_variant for MobileNetV3 Small: {recipe.backbone_config.stem_variant}")
    if recipe.backbone_config.attention_module != "none":
        raise ValueError(f"Unsupported attention_module for MobileNetV3 Small: {recipe.backbone_config.attention_module}")
    if recipe.backbone_config.last_channel_multiplier != 1.0:
        raise ValueError(
            "Unsupported last_channel_multiplier for MobileNetV3 Small: "
            f"{recipe.backbone_config.last_channel_multiplier}"
        )
    if recipe.head_config.classifier_type != "linear":
        raise ValueError(f"Unsupported classifier_type for MobileNetV3 Small: {recipe.head_config.classifier_type}")
    if recipe.components is not None:
        if recipe.components.backbone.name != "mobilenet_v3_small_native":
            raise ValueError(
                f"Unsupported backbone component for MobileNetV3 Small: {recipe.components.backbone.name}"
            )
        if recipe.components.neck.name not in {"avg_pool", "gem_pool"}:
            raise ValueError(f"Unsupported neck component for MobileNetV3 Small: {recipe.components.neck.name}")
        if recipe.components.head.name not in {"native_classifier", "linear", "dropout_linear"}:
            raise ValueError(f"Unsupported head component for MobileNetV3 Small: {recipe.components.head.name}")
    if recipe.modules:
        raise ValueError(f"Unsupported extra recipe modules for MobileNetV3 Small: {sorted(recipe.modules.keys())}")
    stem_layer = recipe.backbone[0]
    if stem_layer.module != "stem_conv":
        raise ValueError("MobileNetV3 Small recipe must start with one stem_conv layer")
    if stem_layer.from_indices != -1:
        raise ValueError("MobileNetV3 Small stem_conv must use from=-1")
    if stem_layer.repeat != 1:
        raise ValueError("MobileNetV3 Small stem_conv must use repeat=1")
    if len(stem_layer.args) != 4:
        raise ValueError("MobileNetV3 Small stem_conv layer requires 4 args")

    for layer in recipe.backbone[1:]:
        if layer.module != "inverted_residual":
            raise ValueError(f"Unsupported backbone module for MobileNetV3 Small: {layer.module}")
        if layer.from_indices != -1:
            raise ValueError("MobileNetV3 Small v1 builder only supports sequential backbone layers")
        if layer.repeat != 1:
            raise ValueError("MobileNetV3 Small v1 builder only supports repeat=1")
        if len(layer.args) != 8:
            raise ValueError("MobileNetV3 Small inverted_residual layers require 8 args")
    allowed_head_modules = {"pointwise_tail", "global_pool", "classifier"}
    for layer in recipe.head:
        if layer.module not in allowed_head_modules:
            raise ValueError(f"Unsupported head module for MobileNetV3 Small: {layer.module}")
        if layer.from_indices != -1:
            raise ValueError("MobileNetV3 Small v1 builder only supports sequential head layers")
        if layer.repeat != 1:
            raise ValueError("MobileNetV3 Small v1 builder only supports repeat=1")
        if layer.module == "pointwise_tail" and len(layer.args) != 2:
            raise ValueError("MobileNetV3 Small pointwise_tail layer requires 2 args")
        if layer.module == "global_pool" and layer.args:
            raise ValueError("MobileNetV3 Small global_pool layer does not accept args")
        if layer.module == "classifier" and len(layer.args) != 1:
            raise ValueError("MobileNetV3 Small classifier layer requires 1 arg")


def _build_mobilenet_v3_small_from_recipe(recipe: ModelRecipe, num_classes: int) -> nn.Module:
    """Build one MobileNetV3 Small model from the supported recipe subset."""
    recipe = hydrate_model_recipe(recipe)
    _validate_mobilenet_v3_small_recipe(recipe)

    output_classes = recipe.nc or num_classes
    backbone_result = build_mobilenet_v3_small_native_backbone(recipe)
    features = list(backbone_result.features)
    tail_module, lastconv_output_channels = build_mobilenet_v3_small_tail(
        recipe,
        input_channels=backbone_result.output_channels,
        norm_layer=backbone_result.norm_layer,
    )
    features.append(tail_module)

    _find_recipe_layer(recipe.head, "global_pool")
    head_component_name = _resolve_head_component_name(recipe)
    neck_component_name = _resolve_neck_component_name(recipe)
    avgpool = build_classification_neck(neck_component_name)
    classifier = build_mobilenet_v3_small_classifier(
        recipe,
        head_name=head_component_name,
        lastconv_output_channels=lastconv_output_channels,
        num_classes=output_classes,
    )
    return RecipeMobileNetV3(
        features=features,
        avgpool=avgpool,
        classifier=classifier,
        num_classes=output_classes,
    )


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


def _validate_resnet18_recipe(recipe: ModelRecipe) -> None:
    """Validate the minimal supported ResNet18 recipe subset."""
    if recipe.base_model != "resnet18":
        raise ValueError(f"Unsupported base_model for ResNet18 builder: {recipe.base_model}")
    if recipe.task_type != "classification":
        raise ValueError(f"Unsupported task_type for ResNet18 builder: {recipe.task_type}")
    if recipe.width_multiple != 1.0:
        raise ValueError("ResNet18 v1 builder does not support width_multiple changes")
    if recipe.backbone_config.stem_variant != "standard":
        raise ValueError(f"Unsupported stem_variant for ResNet18: {recipe.backbone_config.stem_variant}")
    if recipe.backbone_config.attention_module != "none":
        raise ValueError(f"Unsupported attention_module for ResNet18: {recipe.backbone_config.attention_module}")
    if recipe.backbone_config.last_channel_multiplier != 1.0:
        raise ValueError("ResNet18 v1 builder does not support last_channel_multiplier changes")
    if recipe.head_config.pooling_type != "avg":
        raise ValueError(f"Unsupported pooling_type for ResNet18: {recipe.head_config.pooling_type}")
    if recipe.head_config.classifier_type != "linear":
        raise ValueError(f"Unsupported classifier_type for ResNet18: {recipe.head_config.classifier_type}")
    if recipe.head_config.classifier_dropout != 0.0:
        raise ValueError("ResNet18 v1 builder does not support classifier_dropout changes")
    if recipe.components is not None:
        if recipe.components.backbone.name != "resnet18_native":
            raise ValueError(f"Unsupported backbone component for ResNet18: {recipe.components.backbone.name}")
        if recipe.components.neck.name != "avg_pool":
            raise ValueError(f"Unsupported neck component for ResNet18: {recipe.components.neck.name}")
        if recipe.components.head.name != "native_classifier":
            raise ValueError(f"Unsupported head component for ResNet18: {recipe.components.head.name}")
    if recipe.neck:
        raise ValueError("ResNet18 v1 builder does not support neck configuration")
    if recipe.modules:
        raise ValueError(f"Unsupported extra recipe modules for ResNet18: {sorted(recipe.modules.keys())}")
    if recipe.backbone or recipe.head:
        raise ValueError("ResNet18 v1 builder does not support custom architecture layers")


def _build_resnet18_from_recipe(recipe: ModelRecipe, num_classes: int) -> nn.Module:
    """Build one ResNet18 model from the supported recipe subset."""
    _validate_resnet18_recipe(recipe)
    return resnet18(num_classes=recipe.nc or num_classes)


CLASSIFICATION_MODEL_BUILDERS = {
    "mobilenet_v3_small": ClassificationModelBuilderAdapter(
        name="mobilenet_v3_small",
        validate_recipe=_validate_mobilenet_v3_small_recipe,
        build_model=_build_mobilenet_v3_small_from_recipe,
    ),
    "googlenet": ClassificationModelBuilderAdapter(
        name="googlenet",
        validate_recipe=_validate_googlenet_recipe,
        build_model=_build_googlenet_from_recipe,
    ),
    "resnet18": ClassificationModelBuilderAdapter(
        name="resnet18",
        validate_recipe=_validate_resnet18_recipe,
        build_model=_build_resnet18_from_recipe,
    ),
}


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
