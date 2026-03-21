"""Background training runner for the demo backend."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.experiment import ExperimentModel
from app.services.persistence import discard_experiment, get_experiment_config, save_experiment_result, update_experiment_status
from app.trainers.classification.base_trainer import TrainingInterruptedError
from app.workers.experiment_worker import execute_experiment


TRAINING_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="autovisionlab-train")
STOP_EVENTS: dict[str, Event] = {}


def _run_experiment_job(experiment_id: str) -> None:
    """Run one experiment in a background thread."""
    db = SessionLocal()
    try:
        config = get_experiment_config(db, experiment_id)
        if config is None:
            return
        stop_event = STOP_EVENTS.setdefault(experiment_id, Event())
        result = execute_experiment(
            experiment_id=experiment_id,
            config=config,
            should_stop=stop_event.is_set,
        )
        if stop_event.is_set():
            discard_experiment(db, experiment_id)
            return
        save_experiment_result(db, experiment_id=experiment_id, result=result)
    except TrainingInterruptedError:
        discard_experiment(db, experiment_id)
    except Exception:
        update_experiment_status(db, experiment_id, "failed")
    finally:
        STOP_EVENTS.pop(experiment_id, None)
        db.close()


def start_experiment_training(experiment_id: str):
    """Mark an experiment running and schedule background training."""
    db = SessionLocal()
    try:
        experiment = db.get(ExperimentModel, experiment_id)
        if experiment is None:
            return None

        running_experiment = db.scalar(
            select(ExperimentModel).where(
                ExperimentModel.status == "running",
                ExperimentModel.id != experiment_id,
            )
        )
        if running_experiment is not None:
            raise ValueError("Another experiment is already running")

        if experiment.status == "running":
            return experiment

        experiment.status = "running"
        db.commit()
        db.refresh(experiment)
        STOP_EVENTS[experiment_id] = Event()
        TRAINING_EXECUTOR.submit(_run_experiment_job, experiment_id)
        return experiment
    finally:
        db.close()


def stop_experiment_training(experiment_id: str):
    """Request immediate stop for the running experiment and discard it."""
    db = SessionLocal()
    try:
        experiment = db.get(ExperimentModel, experiment_id)
        if experiment is None:
            return None
        if experiment.status != "running":
            raise ValueError("Experiment is not running")

        stop_event = STOP_EVENTS.get(experiment_id)
        if stop_event is None:
            raise ValueError("Experiment stop handle is not available")
        stop_event.set()
        return experiment
    finally:
        db.close()
