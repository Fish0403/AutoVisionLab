"""Background auto-train orchestration for the demo backend."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Lock
import time
from uuid import uuid4

from app.db.session import SessionLocal
from app.schemas.experiment import ExperimentCreateRequest
from app.schemas.parameter_space import SearchPolicy
from app.schemas.run import AutoTrainStartRequest, AutoTrainTaskResponse, RunCreateRequest
from app.services.persistence import create_experiment, create_run, get_experiment_detail, get_run_detail
from app.services.proposal_service import generate_aihubmix_proposal, get_run_history_payload
from app.services.run_policy import (
    get_default_run_policy,
    require_non_basic_change_for_elapsed_budget,
    should_stop_after_dimension_coverage,
)
from app.services.training_runner import start_experiment_training, stop_experiment_training


AUTO_TRAIN_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="autovisionlab-auto-train")
AUTO_TRAIN_TASKS: dict[str, dict] = {}
AUTO_TRAIN_LOCK = Lock()


class AutoTrainStoppedError(RuntimeError):
    """Raised when auto train is stopped by user request."""


def _get_elapsed_seconds(started_at_monotonic: float) -> float:
    """Return elapsed time since the task started."""
    return max(0.0, time.monotonic() - started_at_monotonic)


def _update_elapsed_seconds(task_id: str, started_at_monotonic: float) -> float:
    """Refresh the elapsed time stored in the task snapshot."""
    elapsed_seconds = _get_elapsed_seconds(started_at_monotonic)
    _update_task(task_id, elapsed_seconds=elapsed_seconds)
    return elapsed_seconds


def _append_task_log(task_id: str, message: str) -> None:
    with AUTO_TRAIN_LOCK:
        task = AUTO_TRAIN_TASKS.get(task_id)
        if task is None:
            return
        task["logs"].append(message)
        task["logs"] = task["logs"][-200:]


def _update_task(task_id: str, **updates) -> None:
    with AUTO_TRAIN_LOCK:
        task = AUTO_TRAIN_TASKS.get(task_id)
        if task is None:
            return
        task.update(updates)


def _snapshot_task(task_id: str) -> AutoTrainTaskResponse | None:
    with AUTO_TRAIN_LOCK:
        task = AUTO_TRAIN_TASKS.get(task_id)
        if task is None:
            return None
        payload = deepcopy(task)
    return AutoTrainTaskResponse.model_validate(payload)


def _ensure_no_active_task() -> None:
    with AUTO_TRAIN_LOCK:
        active_task = next(
            (
                task
                for task in AUTO_TRAIN_TASKS.values()
                if task["status"] in {"queued", "running", "stopping"}
            ),
            None,
        )
    if active_task is not None:
        raise ValueError("Another auto train task is already running")


def _build_summary_snapshot(experiment_detail: dict) -> dict:
    metrics = (experiment_detail.get("result") or {}).get("metrics") or {}
    summary_parts = []
    for key in ("top1_acc", "val_loss", "train_loss", "best_epoch"):
        value = metrics.get(key)
        if value is not None:
            summary_parts.append(f"{key}={value}")
    return {
        "experiment_id": experiment_detail["id"],
        "status": experiment_detail["status"],
        "decision": experiment_detail.get("decision"),
        "decision_reason": experiment_detail.get("decision_reason"),
        "metrics": metrics,
        "summary": ", ".join(summary_parts) if summary_parts else "no metrics returned",
    }


def _wait_for_experiment_terminal(task_id: str, experiment_id: str, *, started_at_monotonic: float) -> dict:
    while True:
        _update_elapsed_seconds(task_id, started_at_monotonic)
        task = _snapshot_task(task_id)
        if task is None:
            raise AutoTrainStoppedError("Auto train task disappeared")
        if task.stop_requested:
            try:
                stop_experiment_training(experiment_id)
            except ValueError:
                pass

        db = SessionLocal()
        try:
            experiment_detail = get_experiment_detail(db, experiment_id)
        finally:
            db.close()
        if experiment_detail is None:
            raise ValueError("Experiment not found during auto train")
        detail_payload = experiment_detail.model_dump()
        if detail_payload["status"] in {"success", "failed", "discarded"}:
            return detail_payload
        time.sleep(1)


def _build_followup_config(latest_experiment: dict, proposal_changes: dict) -> dict:
    return {
        "task_type": latest_experiment["config"]["task_type"],
        "dataset": latest_experiment["config"]["dataset"],
        "model_family": latest_experiment["config"]["model_family"],
        "model_name": latest_experiment["config"]["model_name"],
        "parameter_space_version": latest_experiment["config"]["parameter_space_version"],
        "use_demo_mode": latest_experiment["config"].get("use_demo_mode", False),
        "participates_in_ranking": latest_experiment["config"].get("participates_in_ranking", True),
        "search_policy": latest_experiment["config"].get("search_policy") or {},
        "ranking_policy": latest_experiment["config"].get("ranking_policy") or {},
        "params": {
            **latest_experiment["config"]["params"],
            **{key: value for key, value in proposal_changes.items() if value is not None},
        },
    }


def _resolve_followup_source_experiment(db: SessionLocal, run_id: str, proposal_payload: dict) -> dict:
    """Pick the experiment that the next round should branch from."""
    run_detail = get_run_detail(db, run_id)
    if run_detail is None:
        raise ValueError("Run not found during auto train")

    experiment_ids_in_run = {experiment.id for experiment in run_detail.experiments}
    candidate_ids: list[str] = []
    candidate_ids.extend(proposal_payload.get("based_on_experiment_ids") or [])
    if run_detail.frontier_experiment_id:
        candidate_ids.append(run_detail.frontier_experiment_id)
    if run_detail.best_experiment_id:
        candidate_ids.append(run_detail.best_experiment_id)
    if run_detail.experiments:
        candidate_ids.append(run_detail.experiments[-1].id)

    for experiment_id in candidate_ids:
        if experiment_id not in experiment_ids_in_run:
            continue
        experiment_detail = get_experiment_detail(db, experiment_id)
        if experiment_detail is None:
            continue
        return experiment_detail.model_dump()

    raise ValueError("No valid source experiment found for the next auto-train round")


def _load_latest_search_policy_for_run(db: SessionLocal, run_id: str) -> SearchPolicy:
    """Return the latest persisted search policy for one run."""
    run_detail = get_run_detail(db, run_id)
    if run_detail is None or not run_detail.experiments:
        return SearchPolicy()
    latest_experiment_detail = get_experiment_detail(db, run_detail.experiments[-1].id)
    if latest_experiment_detail is None:
        return SearchPolicy()
    experiment_config = latest_experiment_detail.config
    if experiment_config is None:
        return SearchPolicy()
    return experiment_config.search_policy


def _run_auto_train_task(task_id: str, request: AutoTrainStartRequest) -> None:
    try:
        _update_task(task_id, status="running")
        started_at_monotonic = time.monotonic()
        run_policy = get_default_run_policy()
        total_budget_seconds = request.max_wall_clock_minutes * 60
        db = SessionLocal()
        try:
            if request.run_id:
                run_detail = get_run_detail(db, request.run_id)
                if run_detail is None:
                    raise ValueError("Selected run not found")
                run_id = run_detail.id
            else:
                run_detail = create_run(
                    db,
                    RunCreateRequest(
                        name=request.run_name,
                        dataset=request.dataset,
                        model_name=request.model_name,
                        base_config=request.config,
                        notes=None,
                    ),
                )
                run_id = run_detail.id
            _update_task(task_id, run_id=run_id)
        finally:
            db.close()

        if request.run_id:
            db = SessionLocal()
            try:
                parent_run = get_run_detail(db, run_id)
                based_on_experiment_ids = [parent_run.experiments[-1].id] if parent_run and parent_run.experiments else []
            finally:
                db.close()
            baseline_request = ExperimentCreateRequest(
                run_id=run_id,
                config=request.config,
                parameter_space=request.parameter_space,
                proposal={
                    "task_type": "classification",
                    "model_name": request.model_name,
                    "based_on_experiment_ids": based_on_experiment_ids,
                    "hypothesis": "Auto Train baseline appended to the selected run.",
                    "changes": request.config.params.model_dump(),
                    "reason": "Use the current parameter panel as the baseline before AI follow-up rounds.",
                    "risk": "low",
                },
            )
        else:
            baseline_request = ExperimentCreateRequest(
                run_id=run_id,
                config=request.config,
                parameter_space=request.parameter_space,
                proposal=None,
            )

        db = SessionLocal()
        try:
            baseline_experiment = create_experiment(db, baseline_request)
            if baseline_experiment is None:
                raise ValueError("Failed to create baseline experiment")
        finally:
            db.close()
        _update_task(task_id, current_experiment_id=baseline_experiment.id, current_round=0)
        _append_task_log(task_id, f"Baseline created: {baseline_experiment.id}")
        start_experiment_training(baseline_experiment.id)
        _append_task_log(task_id, f"Baseline training started: {baseline_experiment.id}")
        baseline_detail = _wait_for_experiment_terminal(
            task_id,
            baseline_experiment.id,
            started_at_monotonic=started_at_monotonic,
        )
        if baseline_detail["status"] == "discarded":
            raise AutoTrainStoppedError("Baseline experiment was discarded")
        if baseline_detail["status"] != "success":
            raise ValueError(f"Baseline experiment failed with status {baseline_detail['status']}")

        summary = {
            "mode": "auto",
            "run_id": run_id,
            "baseline": _build_summary_snapshot(baseline_detail),
            "rounds": [],
        }
        _update_task(task_id, summary=summary)
        _append_task_log(task_id, f"Baseline finished: {summary['baseline']['summary']}")

        round_index = 0
        consecutive_invalid_proposals = 0
        while True:
            elapsed_seconds = _update_elapsed_seconds(task_id, started_at_monotonic)
            if elapsed_seconds >= total_budget_seconds:
                stop_reason = (
                    f"Stopped by time budget after {elapsed_seconds:.1f}s "
                    f"(budget {request.max_wall_clock_minutes} minutes)."
                )
                _update_task(
                    task_id,
                    status="stopped_by_budget",
                    current_experiment_id=None,
                    stop_reason=stop_reason,
                )
                _append_task_log(task_id, stop_reason)
                return

            db = SessionLocal()
            try:
                search_policy = _load_latest_search_policy_for_run(db, run_id)
                history_payload = get_run_history_payload(db, run_id)
                should_stop, stop_reason = should_stop_after_dimension_coverage(
                    history_payload,
                    search_policy,
                    policy=run_policy,
                )
            finally:
                db.close()

            if should_stop:
                _update_task(
                    task_id,
                    status="stopped_by_policy",
                    current_experiment_id=None,
                    stop_reason=stop_reason,
                )
                _append_task_log(task_id, stop_reason)
                return

            task = _snapshot_task(task_id)
            if task is None or task.stop_requested:
                raise AutoTrainStoppedError("Auto train stopped by user request")
            round_index += 1
            _update_task(task_id, current_round=round_index)
            require_non_basic_change = require_non_basic_change_for_elapsed_budget(
                elapsed_seconds,
                request.max_wall_clock_minutes,
                policy=run_policy,
            )
            if require_non_basic_change:
                _append_task_log(
                    task_id,
                    f"Round {round_index}: stage policy requires at least one augmentation/loss/strategy change",
                )
            _append_task_log(task_id, f"Round {round_index}: generating AI proposal")

            db = SessionLocal()
            try:
                try:
                    proposal = generate_aihubmix_proposal(
                        db,
                        run_id,
                        require_non_basic_change=require_non_basic_change,
                    )
                except ValueError as error:
                    consecutive_invalid_proposals += 1
                    _append_task_log(
                        task_id,
                        f"Round {round_index}: invalid proposal ({consecutive_invalid_proposals} consecutive) | {error}",
                    )
                    continue
            finally:
                db.close()
            consecutive_invalid_proposals = 0

            proposal_payload = proposal.model_dump()
            _append_task_log(task_id, f"Round {round_index}: AI suggested {proposal.hypothesis}")

            db = SessionLocal()
            try:
                source_experiment_detail = _resolve_followup_source_experiment(db, run_id, proposal_payload)
            finally:
                db.close()
            _append_task_log(
                task_id,
                f"Round {round_index}: branching from {source_experiment_detail['id']} for the next experiment",
            )

            followup_config = _build_followup_config(source_experiment_detail, proposal_payload["changes"])
            db = SessionLocal()
            try:
                next_experiment = create_experiment(
                    db,
                    ExperimentCreateRequest(
                        run_id=run_id,
                        config=followup_config,
                        parameter_space=request.parameter_space,
                        proposal=proposal,
                    ),
                )
                if next_experiment is None:
                    raise ValueError("Failed to create follow-up experiment")
            finally:
                db.close()

            _update_task(task_id, current_experiment_id=next_experiment.id)
            _append_task_log(task_id, f"Round {round_index}: created experiment {next_experiment.id}")
            start_experiment_training(next_experiment.id)
            _append_task_log(task_id, f"Round {round_index}: training started for {next_experiment.id}")
            current_experiment_detail = _wait_for_experiment_terminal(
                task_id,
                next_experiment.id,
                started_at_monotonic=started_at_monotonic,
            )
            if current_experiment_detail["status"] == "discarded":
                raise AutoTrainStoppedError("Current experiment was discarded")
            if current_experiment_detail["status"] != "success":
                raise ValueError(f"Round {round_index} experiment failed with status {current_experiment_detail['status']}")

            summary["rounds"].append(
                {
                    "round_index": round_index,
                    "proposal": proposal_payload,
                    "result": _build_summary_snapshot(current_experiment_detail),
                }
            )
            _update_task(task_id, summary=summary)
            _append_task_log(task_id, f"Round {round_index}: {summary['rounds'][-1]['result']['summary']}")
            if current_experiment_detail.get("decision") == "keep":
                _append_task_log(
                    task_id,
                    f"Round {round_index}: promoted to the active frontier | "
                    f"{current_experiment_detail.get('decision_reason')}",
                )
            else:
                _append_task_log(
                    task_id,
                    f"Round {round_index}: no promotion, revert to current best/frontier | "
                    f"{current_experiment_detail.get('decision_reason')}",
                )
    except AutoTrainStoppedError:
        _update_task(task_id, status="stopped", current_experiment_id=None, stop_reason="Stopped by user request.")
        _append_task_log(task_id, "Auto train stopped and current experiment discarded")
    except Exception as error:
        _update_task(task_id, status="failed", error=str(error), current_experiment_id=None)
        _append_task_log(task_id, f"Auto train failed: {error}")
        error_detail = getattr(error, "response", None)
        if error_detail is not None and getattr(error_detail, "text", None):
            _append_task_log(task_id, f"Provider detail: {error_detail.text}")


def start_auto_train_task(request: AutoTrainStartRequest) -> AutoTrainTaskResponse:
    """Start one background auto-train task."""
    _ensure_no_active_task()
    task_id = f"auto_{uuid4().hex[:8]}"
    task_payload = {
        "task_id": task_id,
        "status": "queued",
        "run_id": request.run_id,
        "current_round": 0,
        "max_wall_clock_minutes": request.max_wall_clock_minutes,
        "elapsed_seconds": 0.0,
        "current_experiment_id": None,
        "logs": [],
        "summary": None,
        "error": None,
        "stop_requested": False,
        "stop_reason": None,
    }
    with AUTO_TRAIN_LOCK:
        AUTO_TRAIN_TASKS[task_id] = task_payload
    AUTO_TRAIN_EXECUTOR.submit(_run_auto_train_task, task_id, request)
    return AutoTrainTaskResponse.model_validate(task_payload)


def get_auto_train_task(task_id: str) -> AutoTrainTaskResponse | None:
    """Return one auto-train task snapshot."""
    return _snapshot_task(task_id)


def stop_auto_train_task(task_id: str) -> AutoTrainTaskResponse | None:
    """Request stop for one running auto-train task."""
    with AUTO_TRAIN_LOCK:
        task = AUTO_TRAIN_TASKS.get(task_id)
        if task is None:
            return None
        task["stop_requested"] = True
        if task["status"] in {"queued", "running"}:
            task["status"] = "stopping"
        current_experiment_id = task.get("current_experiment_id")
    if current_experiment_id:
        try:
            stop_experiment_training(current_experiment_id)
        except ValueError:
            pass
    return _snapshot_task(task_id)
