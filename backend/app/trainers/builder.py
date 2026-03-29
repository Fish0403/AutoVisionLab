"""Unified builder entrypoints for trainer-side model construction."""

from __future__ import annotations

from torch import nn

from app.schemas.parameter_space import ExperimentConfig, ModelRecipe
from app.trainers.classification.model_builder_registry import build_classification_model_from_recipe
from app.trainers.manifest import TrainerManifest
from app.trainers.validator import validate_model_recipe, validate_trainer_manifest


def build_model_from_recipe(recipe: ModelRecipe, *, num_classes: int = 10) -> nn.Module:
    """Build one model directly from one validated model recipe."""
    validate_model_recipe(recipe)
    return build_classification_model_from_recipe(recipe, num_classes=num_classes)


def _resolve_output_classes(manifest: TrainerManifest) -> int:
    """Resolve the output class count from manifest data with a stable fallback."""
    if manifest.model.nc is not None:
        return manifest.model.nc
    if manifest.data.class_names:
        return len(manifest.data.class_names)
    return 10


def build_model_from_manifest(manifest: TrainerManifest) -> nn.Module:
    """Build one model from the unified trainer manifest."""
    validate_trainer_manifest(manifest)
    return build_model_from_recipe(manifest.model, num_classes=_resolve_output_classes(manifest))


def build_model_from_config(config: ExperimentConfig) -> nn.Module:
    """Build one model from the persisted experiment config entrypoint."""
    manifest = TrainerManifest.from_experiment_config(config)
    return build_model_from_manifest(manifest)
