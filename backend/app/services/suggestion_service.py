"""Background suggestion generation for completed experiments."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Lock
from uuid import uuid4

from app.db.session import SessionLocal
from app.schemas.experiment import ExperimentSuggestionTaskResponse
from app.services.persistence import get_experiment_detail
from app.services.proposal_service import generate_aihubmix_proposal


SUGGESTION_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="autovisionlab-suggestion")
SUGGESTION_LOCK = Lock()
SUGGESTION_TASKS: dict[str, dict] = {}
LATEST_SUGGESTION_TASK_BY_EXPERIMENT: dict[str, str] = {}


def _update_suggestion_task(task_id: str, **updates) -> None:
    with SUGGESTION_LOCK:
        task = SUGGESTION_TASKS.get(task_id)
        if task is None:
            return
        task.update(updates)


def _snapshot_suggestion_task(task_id: str) -> ExperimentSuggestionTaskResponse | None:
    with SUGGESTION_LOCK:
        task = SUGGESTION_TASKS.get(task_id)
        if task is None:
            return None
        payload = deepcopy(task)
    return ExperimentSuggestionTaskResponse.model_validate(payload)


def _record_provider_usage(task_id: str, provider_metadata: dict) -> None:
    """Store provider token usage for one suggestion task."""
    usage = provider_metadata.get("usage") or {}
    _update_suggestion_task(
        task_id,
        latest_provider_prompt_tokens=int(usage.get("prompt_tokens") or 0) or None,
        latest_provider_completion_tokens=int(usage.get("completion_tokens") or 0) or None,
        latest_provider_total_tokens=int(usage.get("total_tokens") or 0) or None,
    )


def _run_suggestion_task(task_id: str, experiment_id: str, run_id: str) -> None:
    _update_suggestion_task(task_id, status="running")
    db = SessionLocal()
    try:
        experiment_detail = get_experiment_detail(db, experiment_id)
        if experiment_detail is None:
            raise ValueError("Experiment not found")
        if experiment_detail.status not in {"success", "failed", "discarded"}:
            raise ValueError("Suggestion generation requires one completed experiment")
        suggestion = generate_aihubmix_proposal(
            db,
            run_id,
            on_provider_metadata=lambda provider_metadata: _record_provider_usage(task_id, provider_metadata),
        )
        _update_suggestion_task(task_id, status="success", suggestion=suggestion.model_dump())
    except Exception as error:
        _update_suggestion_task(task_id, status="failed", error=str(error))
    finally:
        db.close()


def start_experiment_suggestion_task(experiment_id: str) -> ExperimentSuggestionTaskResponse:
    """Start or reuse one background suggestion task for the given experiment."""
    db = SessionLocal()
    try:
        experiment_detail = get_experiment_detail(db, experiment_id)
        if experiment_detail is None:
            raise ValueError("Experiment not found")
        if experiment_detail.status not in {"success", "failed", "discarded"}:
            raise ValueError("Suggestion generation requires one completed experiment")
        run_id = experiment_detail.run_id
    finally:
        db.close()

    with SUGGESTION_LOCK:
        existing_task_id = LATEST_SUGGESTION_TASK_BY_EXPERIMENT.get(experiment_id)
        if existing_task_id is not None:
            existing_task = SUGGESTION_TASKS.get(existing_task_id)
            if existing_task is not None and existing_task.get("status") in {"queued", "running", "success"}:
                return ExperimentSuggestionTaskResponse.model_validate(deepcopy(existing_task))

        task_id = f"suggest_{uuid4().hex[:8]}"
        task_payload = {
            "task_id": task_id,
            "experiment_id": experiment_id,
            "run_id": run_id,
            "status": "queued",
            "suggestion": None,
            "error": None,
            "latest_provider_prompt_tokens": None,
            "latest_provider_completion_tokens": None,
            "latest_provider_total_tokens": None,
        }
        SUGGESTION_TASKS[task_id] = task_payload
        LATEST_SUGGESTION_TASK_BY_EXPERIMENT[experiment_id] = task_id

    SUGGESTION_EXECUTOR.submit(_run_suggestion_task, task_id, experiment_id, run_id)
    return ExperimentSuggestionTaskResponse.model_validate(task_payload)


def get_experiment_suggestion_task(experiment_id: str) -> ExperimentSuggestionTaskResponse | None:
    """Return the latest suggestion task for one experiment when available."""
    with SUGGESTION_LOCK:
        task_id = LATEST_SUGGESTION_TASK_BY_EXPERIMENT.get(experiment_id)
    if task_id is None:
        return None
    return _snapshot_suggestion_task(task_id)
