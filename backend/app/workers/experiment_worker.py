"""Synchronous experiment execution for the demo backend."""

from typing import Callable

from app.schemas.ai import ResultSchema
from app.schemas.parameter_space import ExperimentConfig
from app.trainers.factory import build_trainer_from_config


def execute_experiment(
    experiment_id: str,
    run_id: str,
    config: ExperimentConfig,
    should_stop: Callable[[], bool] | None = None,
) -> ResultSchema:
    """Execute one experiment synchronously."""
    trainer = build_trainer_from_config(
        config,
        experiment_id=experiment_id,
        run_id=run_id,
        should_stop=should_stop,
    )
    return trainer.train()
