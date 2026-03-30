"""Unified builder entrypoints for trainer-side model construction."""

from __future__ import annotations

from pathlib import Path

from torch import nn

from app.schemas.parameter_space import ExperimentConfig, ModelRecipe
from app.services.dataset_paths import resolve_dataset_dir
from app.trainers.classification.data_loading import collect_manifest_classes
from app.trainers.classification.model_builder_registry import build_classification_model_from_recipe
from app.trainers.manifest import TrainerManifest
from app.trainers.validator import validate_model_recipe, validate_trainer_manifest


def build_model_from_recipe(recipe: ModelRecipe, *, num_classes: int = 10) -> nn.Module:
    """Build one model directly from one validated model recipe."""
    validate_model_recipe(recipe)
    return build_classification_model_from_recipe(recipe, num_classes=num_classes)


def _infer_output_classes_from_manifest_splits(manifest: TrainerManifest) -> int | None:
    """Infer class count from manifest split files when class_names are not populated."""
    manifest_paths = [
        Path(manifest.data.splits.train_manifest),
        Path(manifest.data.splits.val_manifest),
    ]

    class_names: set[str] = set()
    for manifest_path in manifest_paths:
        if not manifest_path.exists() and len(manifest_path.parts) >= 3:
            candidate_parent = resolve_dataset_dir(
                manifest_path.parent.parent,
                manifest.data.dataset_name,
                strict=False,
            )
            candidate_path = candidate_parent / manifest_path.name
            if candidate_path.exists():
                manifest_path = candidate_path
        if not manifest_path.exists():
            continue
        class_names.update(collect_manifest_classes(manifest_path))
    if not class_names:
        return None
    return len(class_names)


def _resolve_output_classes(manifest: TrainerManifest) -> int:
    """Resolve the output class count from manifest data with a stable fallback."""
    if manifest.model.nc is not None:
        return manifest.model.nc
    if manifest.data.class_names:
        return len(manifest.data.class_names)
    inferred_output_classes = _infer_output_classes_from_manifest_splits(manifest)
    if inferred_output_classes is not None:
        return inferred_output_classes
    return 10


def build_model_from_manifest(manifest: TrainerManifest) -> nn.Module:
    """Build one model from the unified trainer manifest."""
    validate_trainer_manifest(manifest)
    return build_model_from_recipe(manifest.model, num_classes=_resolve_output_classes(manifest))


def build_model_from_config(config: ExperimentConfig) -> nn.Module:
    """Build one model from the persisted experiment config entrypoint."""
    manifest = TrainerManifest.from_experiment_config(config)
    return build_model_from_manifest(manifest)
