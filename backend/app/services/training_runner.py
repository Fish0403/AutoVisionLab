"""Background training runner for the demo backend."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.experiment import ExperimentModel
from app.services.log_events import format_run_log_message
from app.services.persistence import discard_experiment, get_experiment_config, save_experiment_result, update_experiment_status
from app.services.run_logging import append_run_log
from app.trainers.classification.base_trainer import TrainingInterruptedError
from app.workers.experiment_worker import execute_experiment


TRAINING_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="autovisionlab-train")
STOP_EVENTS: dict[str, Event] = {}


def is_cuda_oom_error(error: Exception) -> bool:
    """Return whether one exception represents a CUDA OOM failure."""
    error_message = str(error).lower()
    return "cuda out of memory" in error_message or "cuda error: out of memory" in error_message


def build_failure_reason(error: Exception) -> str:
    """Build a user-facing failure reason while preserving the raw error context."""
    raw_message = str(error)
    if is_cuda_oom_error(error):
        return (
            "CUDA out of memory. Try a smaller batch size, "
            "or reduce image size if needed. "
            f"Raw error: {raw_message}"
        )
    return raw_message


def cleanup_stale_running_experiments(db: Session) -> int:
    """Discard experiments left in running state after an unclean shutdown."""
    stale_experiments = db.scalars(select(ExperimentModel).where(ExperimentModel.status == "running")).all()
    for experiment in stale_experiments:
        discard_experiment(db, experiment.id)
    STOP_EVENTS.clear()
    return len(stale_experiments)


def _run_experiment_job(experiment_id: str) -> None:
    """Run one experiment in a background thread."""
    db = SessionLocal()
    try:
        config = get_experiment_config(db, experiment_id)
        if config is None:
            return
        experiment = db.get(ExperimentModel, experiment_id)
        if experiment is None:
            return
        run_id = experiment.run_id
        stop_event = STOP_EVENTS.setdefault(experiment_id, Event())
        result = execute_experiment(
            experiment_id=experiment_id,
            run_id=run_id,
            config=config,
            should_stop=stop_event.is_set,
        )
        if stop_event.is_set():
            append_run_log(
                run_id,
                format_run_log_message(
                    level="WARNING",
                    section="training",
                    message="stop requested; experiment discarded",
                    experiment_id=experiment_id,
                ),
            )
            discard_experiment(db, experiment_id)
            return
        save_experiment_result(db, experiment_id=experiment_id, result=result)
    except TrainingInterruptedError:
        experiment = db.get(ExperimentModel, experiment_id)
        if experiment is not None:
            append_run_log(
                experiment.run_id,
                format_run_log_message(
                    level="WARNING",
                    section="training",
                    message="training interrupted and discarded",
                    experiment_id=experiment_id,
                ),
            )
        discard_experiment(db, experiment_id)
    except Exception as error:
        update_experiment_status(db, experiment_id, "failed")
        experiment = db.get(ExperimentModel, experiment_id)
        if experiment is not None:
            append_run_log(
                experiment.run_id,
                format_run_log_message(
                    level="ERROR",
                    section="training",
                    message="training failed",
                    experiment_id=experiment_id,
                    error=str(error),
                ),
            )
            experiment.decision = "crash"
            experiment.decision_reason = build_failure_reason(error)
            db.commit()
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
        append_run_log(
            experiment.run_id,
            format_run_log_message(
                level="INFO",
                section="training",
                message="training started",
                experiment_id=experiment_id,
            ),
        )
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
