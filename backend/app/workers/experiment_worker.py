"""Synchronous experiment execution for the demo backend."""

from typing import Callable

from app.schemas.ai import ResultSchema
from app.schemas.parameter_space import ExperimentConfig
from app.trainers.classification.densenet_trainer import DenseNet121Trainer
from app.trainers.classification.googlenet_trainer import GoogLeNetTrainer
from app.trainers.classification.mobilenet_trainer import MobileNetTrainer
from app.trainers.classification.resnet_trainer import ResNet18Trainer, ResNet34Trainer


def execute_experiment(
    experiment_id: str,
    run_id: str,
    config: ExperimentConfig,
    should_stop: Callable[[], bool] | None = None,
) -> ResultSchema:
    """Execute one experiment synchronously."""
    if config.model_name == "mobilenet_v2":
        return MobileNetTrainer(config=config, experiment_id=experiment_id, run_id=run_id, should_stop=should_stop).train()
    if config.model_name == "googlenet":
        return GoogLeNetTrainer(config=config, experiment_id=experiment_id, run_id=run_id, should_stop=should_stop).train()
    if config.model_name == "resnet18":
        return ResNet18Trainer(config=config, experiment_id=experiment_id, run_id=run_id, should_stop=should_stop).train()
    if config.model_name == "resnet34":
        return ResNet34Trainer(config=config, experiment_id=experiment_id, run_id=run_id, should_stop=should_stop).train()
    if config.model_name == "densenet121":
        return DenseNet121Trainer(config=config, experiment_id=experiment_id, run_id=run_id, should_stop=should_stop).train()
    raise ValueError(f"Unsupported model_name: {config.model_name}")
