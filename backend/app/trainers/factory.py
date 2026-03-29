"""Unified trainer factory entrypoint."""

from __future__ import annotations

from typing import Callable

from app.schemas.parameter_space import ExperimentConfig
from app.trainers.classification.base_trainer import BaseClassificationTrainer
from app.trainers.classification.model_adapters import get_classification_model_adapter
from app.trainers.manifest import TrainerManifest
from app.trainers.validator import validate_trainer_manifest


def build_trainer_from_config(
    config: ExperimentConfig,
    *,
    experiment_id: str,
    run_id: str,
    should_stop: Callable[[], bool] | None = None,
) -> BaseClassificationTrainer:
    """Instantiate one validated trainer from the persisted config object."""
    manifest = TrainerManifest.from_experiment_config(config)
    validate_trainer_manifest(manifest)
    model_adapter = get_classification_model_adapter(manifest.model_name)
    return BaseClassificationTrainer(
        config=config,
        experiment_id=experiment_id,
        run_id=run_id,
        should_stop=should_stop,
        model_adapter=model_adapter,
    )
