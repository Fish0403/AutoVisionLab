"""Synchronous experiment execution for the demo backend."""

from typing import Callable

from app.schemas.ai import ResultSchema
from app.schemas.parameter_space import ExperimentConfig
from app.trainers.classification.googlenet_trainer import GoogLeNetTrainer
from app.trainers.classification.mobilenet_trainer import MobileNetTrainer


def execute_experiment(
    experiment_id: str,
    config: ExperimentConfig,
    should_stop: Callable[[], bool] | None = None,
) -> ResultSchema:
    """Execute one experiment synchronously."""
    if config.model_name == "mobilenet_v2":
        return MobileNetTrainer(config=config, experiment_id=experiment_id, should_stop=should_stop).train()
    if config.model_name == "googlenet":
        return GoogLeNetTrainer(config=config, experiment_id=experiment_id, should_stop=should_stop).train()
    raise ValueError(f"Unsupported model_name: {config.model_name}")
