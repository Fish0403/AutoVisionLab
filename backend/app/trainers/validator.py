"""Unified validator entrypoints for trainer manifests and model recipes."""

from __future__ import annotations

from app.schemas.parameter_space import DatasetRecipe, ModelRecipe
from app.trainers.classification.model_builder_registry import validate_classification_model_recipe
from app.trainers.manifest import TrainerManifest


def validate_model_recipe(recipe: ModelRecipe) -> None:
    """Validate one structured model recipe with the family-specific validator."""
    validate_classification_model_recipe(recipe)


def validate_dataset_recipe(dataset_recipe: DatasetRecipe) -> None:
    """Validate the dataset block required by the current trainer pipeline."""
    if dataset_recipe.task_type != "classification":
        raise ValueError(f"Unsupported dataset task_type: {dataset_recipe.task_type}")
    if not dataset_recipe.dataset_name.strip():
        raise ValueError("Dataset recipe dataset_name must not be empty")
    if not dataset_recipe.splits.train_manifest.strip():
        raise ValueError("Dataset recipe train_manifest must not be empty")
    if not dataset_recipe.splits.val_manifest.strip():
        raise ValueError("Dataset recipe val_manifest must not be empty")
    if dataset_recipe.metadata.image_size_options and dataset_recipe.metadata.image_size_options != sorted(
        dataset_recipe.metadata.image_size_options
    ):
        raise ValueError("Dataset recipe image_size_options must be sorted when provided")


def validate_trainer_manifest(manifest: TrainerManifest) -> None:
    """Validate the unified manifest before training or model building."""
    if manifest.task_type != "classification":
        raise ValueError(f"Unsupported trainer manifest task_type: {manifest.task_type}")
    validate_model_recipe(manifest.model)
    validate_dataset_recipe(manifest.data)
    if (
        manifest.data.metadata.image_size_options
        and manifest.train.image_size not in manifest.data.metadata.image_size_options
    ):
        raise ValueError(
            "train.image_size must be declared in data.metadata.image_size_options when options are provided"
        )
