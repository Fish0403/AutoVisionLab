"""Helpers for building frontend-facing model config defaults."""

from __future__ import annotations

from app.model_catalog.registry import get_model_catalog_entry
from app.model_catalog.schemas import ModelDefaults, ModelSummary
from app.schemas.parameter_space import (
    DiscreteValuesParamDefinition,
    EditableParameterSpace,
    EnumParamDefinition,
    NumberRangeParamDefinition,
    SearchPolicy,
    TrainHyp,
)
from app.schemas.ranking_policy import RankingPolicy


DEFAULT_IMAGE_SIZE = 224
DEFAULT_BASIC_SEARCH_FIELDS = [
    "optimizer",
    "learning_rate",
    "batch_size",
    "weight_decay",
    "scheduler",
    "label_smoothing",
    "image_size",
]
LOSS_SEARCH_FIELDS = {"loss_name", "focal_gamma"}
AUGMENTATION_SEARCH_FIELDS = {"mixup_alpha", "cutmix_alpha", "random_erasing_prob"}
MODEL_MODULE_SEARCH_FIELDS = {"aux_logits", "neck_name", "head_name"}


def _pick_enum_choice(
    parameter_space: EditableParameterSpace,
    field_name: str,
    preferred_values: list[str],
    *,
    fallback: str,
) -> str:
    """Return the preferred enum choice when one is declared."""
    definition = parameter_space.editable_params.get(field_name)
    if not isinstance(definition, EnumParamDefinition):
        return fallback
    normalized_choices = [str(choice) for choice in definition.choices]
    for preferred_value in preferred_values:
        if preferred_value in normalized_choices:
            return preferred_value
    return normalized_choices[0] if normalized_choices else fallback


def _pick_discrete_number(
    parameter_space: EditableParameterSpace,
    field_name: str,
    preferred_values: list[int],
    *,
    fallback: int,
) -> int:
    """Return the preferred discrete numeric choice when one is declared."""
    definition = parameter_space.editable_params.get(field_name)
    if not isinstance(definition, DiscreteValuesParamDefinition):
        return fallback
    numeric_choices = [int(choice) for choice in definition.choices]
    for preferred_value in preferred_values:
        if preferred_value in numeric_choices:
            return preferred_value
    return numeric_choices[0] if numeric_choices else fallback


def _pick_ranged_number(
    parameter_space: EditableParameterSpace,
    field_name: str,
    preferred_value: float,
    *,
    fallback: float,
) -> float:
    """Clamp the preferred numeric value into the declared range when needed."""
    definition = parameter_space.editable_params.get(field_name)
    if not isinstance(definition, NumberRangeParamDefinition):
        return fallback
    return min(max(preferred_value, definition.min), definition.max)


def build_default_train_hyp(parameter_space: EditableParameterSpace) -> TrainHyp:
    """Build one default train_hyp template that respects the declared parameter space."""
    optimizer = _pick_enum_choice(
        parameter_space,
        "optimizer",
        ["adamw", "adam", "sgd"],
        fallback="adamw",
    )
    scheduler = _pick_enum_choice(
        parameter_space,
        "scheduler",
        ["cosine", "step", "none"],
        fallback="cosine",
    )
    loss_name = _pick_enum_choice(
        parameter_space,
        "loss_name",
        ["cross_entropy_with_label_smoothing", "cross_entropy", "focal_loss"],
        fallback="cross_entropy_with_label_smoothing",
    )
    learning_rate = _pick_ranged_number(
        parameter_space,
        "learning_rate",
        0.003,
        fallback=0.003,
    )
    weight_decay = _pick_ranged_number(
        parameter_space,
        "weight_decay",
        0.0001,
        fallback=0.0001,
    )
    label_smoothing = _pick_ranged_number(
        parameter_space,
        "label_smoothing",
        0.1,
        fallback=0.1,
    )
    focal_gamma = _pick_ranged_number(
        parameter_space,
        "focal_gamma",
        2.0,
        fallback=2.0,
    )
    mixup_alpha = _pick_ranged_number(
        parameter_space,
        "mixup_alpha",
        0.0,
        fallback=0.0,
    )
    cutmix_alpha = _pick_ranged_number(
        parameter_space,
        "cutmix_alpha",
        0.0,
        fallback=0.0,
    )
    random_erasing_prob = _pick_ranged_number(
        parameter_space,
        "random_erasing_prob",
        0.0,
        fallback=0.0,
    )
    image_size = int(
        round(
            _pick_ranged_number(
                parameter_space,
                "image_size",
                DEFAULT_IMAGE_SIZE,
                fallback=DEFAULT_IMAGE_SIZE,
            )
        )
    )
    batch_size = _pick_discrete_number(
        parameter_space,
        "batch_size",
        [64, 32, 128, 16],
        fallback=64,
    )
    epochs = _pick_discrete_number(
        parameter_space,
        "epochs",
        [10, 20, 30, 50],
        fallback=10,
    )

    return TrainHyp.model_validate(
        {
            "version": "train_hyp@v1",
            "task_type": "classification",
            "optimizer": optimizer,
            "lr0": learning_rate,
            "weight_decay": weight_decay,
            "scheduler": scheduler,
            "epochs": epochs,
            "batch_size": batch_size,
            "image_size": image_size,
            "label_smoothing": label_smoothing,
            "fl_gamma": focal_gamma if loss_name == "focal_loss" else 0.0,
            "augmentation": {
                "mixup": mixup_alpha,
                "cutmix": cutmix_alpha,
                "random_erasing": random_erasing_prob,
            },
            "loss": {
                "name": loss_name,
            },
        }
    )


def build_default_search_policy(parameter_space: EditableParameterSpace) -> SearchPolicy:
    """Build one default search policy aligned with the declared editable fields."""
    editable_fields = set(parameter_space.editable_params.keys())
    allowed_basic_hparam_fields = [
        field_name for field_name in DEFAULT_BASIC_SEARCH_FIELDS if field_name in editable_fields
    ]
    return SearchPolicy(
        allow_basic_hparam_search=bool(allowed_basic_hparam_fields),
        allowed_basic_hparam_fields=allowed_basic_hparam_fields,
        allow_strategy_search=False,
        allow_loss_search=bool(editable_fields & LOSS_SEARCH_FIELDS),
        allow_augmentation_search=bool(editable_fields & AUGMENTATION_SEARCH_FIELDS),
        allow_model_module_search=bool(editable_fields & MODEL_MODULE_SEARCH_FIELDS),
        require_manual_approval_for_high_impact_changes=True,
    )


def build_default_ranking_policy() -> RankingPolicy:
    """Build the workspace default ranking policy."""
    return RankingPolicy(
        primary_metric="top1_acc",
        primary_metric_mode="max",
        min_primary_metric_improvement=0.001,
        primary_metric_parity_epsilon=0.0005,
        tie_breaker_metric="latency_ms",
        tie_breaker_mode="min",
        min_tie_breaker_metric_improvement=0.5,
        max_image_size=None,
    )


def get_model_defaults(model_name: str) -> ModelDefaults | None:
    """Return the default config template for one registered model."""
    model_entry = get_model_catalog_entry(model_name)
    if model_entry is None:
        return None
    parameter_space = model_entry.parameter_space.model_copy(deep=True)
    return ModelDefaults(
        summary=ModelSummary.from_entry(model_entry),
        parameter_space=parameter_space,
        default_model_recipe=model_entry.default_model_recipe.model_copy(deep=True),
        default_train_hyp=build_default_train_hyp(parameter_space),
        default_search_policy=build_default_search_policy(parameter_space),
        default_ranking_policy=build_default_ranking_policy(),
    )
