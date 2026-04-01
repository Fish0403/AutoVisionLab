"""Background auto-train orchestration for the demo backend."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock
import time
from uuid import uuid4

import requests

from app.core.settings import get_settings
from app.db.session import SessionLocal
from app.llm.aihubmix_client import AIHubMixRequestError
from app.schemas.experiment import ExperimentCreateRequest
from app.schemas.parameter_space import (
    ExperimentConfig,
    SearchPolicy,
    apply_model_recipe_change_payload,
    apply_proposal_changes_to_model_recipe,
    apply_train_hyp_change_payload,
    apply_proposal_changes_to_train_hyp,
    build_model_recipe_change_payload,
    build_train_hyp_change_payload,
)
from app.schemas.run import AutoTrainStartRequest, AutoTrainTaskResponse, RunCreateRequest, TaskHistoryItemResponse
from app.services.dataset_service import build_dataset_summary_text, get_local_dataset_summary
from app.services.persistence import clear_run_records, create_experiment, create_run, get_experiment_detail, get_run_detail
from app.services.proposal_service import generate_aihubmix_proposal, get_run_history_payload
from app.services.run_policy import (
    get_default_run_policy,
    require_non_basic_change_after_warmup_rounds,
    should_stop_after_dimension_coverage,
)
from app.services.task_store import delete_task_payload, get_active_task_payload, get_task_payload, list_task_payloads, upsert_task_payload
from app.services.training_runner import start_experiment_training, stop_experiment_training


AUTO_TRAIN_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="autovisionlab-auto-train")
AUTO_TRAIN_TASKS: dict[str, dict] = {}
AUTO_TRAIN_LOCK = Lock()
AUTO_TRAIN_PROPOSAL_RETRY_DELAYS_SECONDS = (3, 6, 10)
AUTO_TRAIN_INVALID_PROPOSAL_RETRY_DELAY_SECONDS = 2
AUTO_TRAIN_MAX_CONSECUTIVE_INVALID_PROPOSALS = 3


class AutoTrainStoppedError(RuntimeError):
    """Raised when auto train is stopped by user request."""


def _get_elapsed_seconds(started_at_monotonic: float) -> float:
    """Return elapsed time since the task started."""
    return max(0.0, time.monotonic() - started_at_monotonic)


def _now_iso() -> str:
    """Return one UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


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
        task["updated_at"] = _now_iso()
        payload = deepcopy(task)
    upsert_task_payload("auto_train", payload)


def _record_prompt_token_estimate(task_id: str, prompt_metadata: dict) -> None:
    """Update task-level token counters from one proposal prompt estimate."""
    prompt_tokens_estimate = int(prompt_metadata.get("prompt_tokens_estimate") or 0)
    history_items = prompt_metadata.get("history_items")
    with AUTO_TRAIN_LOCK:
        task = AUTO_TRAIN_TASKS.get(task_id)
        if task is None:
            return
        task["latest_prompt_tokens_estimate"] = prompt_tokens_estimate
        task["estimated_prompt_tokens_total"] = int(task.get("estimated_prompt_tokens_total") or 0) + prompt_tokens_estimate
        task["latest_prompt_history_items"] = int(history_items) if isinstance(history_items, int) else None
        payload = deepcopy(task)
    upsert_task_payload("auto_train", payload)


def _record_provider_usage(task_id: str, provider_metadata: dict) -> None:
    """Update task-level token counters from one provider usage payload."""
    usage = provider_metadata.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or 0)
    with AUTO_TRAIN_LOCK:
        task = AUTO_TRAIN_TASKS.get(task_id)
        if task is None:
            return
        task["latest_provider_prompt_tokens"] = prompt_tokens or None
        task["latest_provider_completion_tokens"] = completion_tokens or None
        task["latest_provider_total_tokens"] = total_tokens or None
        task["provider_prompt_tokens_total"] = int(task.get("provider_prompt_tokens_total") or 0) + prompt_tokens
        task["provider_completion_tokens_total"] = int(task.get("provider_completion_tokens_total") or 0) + completion_tokens
        task["provider_total_tokens_total"] = int(task.get("provider_total_tokens_total") or 0) + total_tokens
        payload = deepcopy(task)
    upsert_task_payload("auto_train", payload)


def _append_owned_run(task_id: str, run_id: str) -> None:
    """Attach one task-owned run identifier for later cleanup."""
    with AUTO_TRAIN_LOCK:
        task = AUTO_TRAIN_TASKS.get(task_id)
        if task is not None:
            owned_run_ids = list(task.get("owned_run_ids") or [])
            if run_id not in owned_run_ids:
                owned_run_ids.append(run_id)
            task["owned_run_ids"] = owned_run_ids
            task["updated_at"] = _now_iso()
            payload = deepcopy(task)
        else:
            payload = get_task_payload("auto_train", task_id)
            if payload is None:
                return
            owned_run_ids = list(payload.get("owned_run_ids") or [])
            if run_id not in owned_run_ids:
                owned_run_ids.append(run_id)
            payload["owned_run_ids"] = owned_run_ids
            payload["updated_at"] = _now_iso()
    upsert_task_payload("auto_train", payload)


def _set_activity_message(task_id: str, activity_message: str | None) -> None:
    """Update one short task activity message for the workspace."""
    _update_task(task_id, activity_message=activity_message)


def _update_task(task_id: str, **updates) -> None:
    with AUTO_TRAIN_LOCK:
        task = AUTO_TRAIN_TASKS.get(task_id)
        if task is not None:
            task.update(updates)
            task["updated_at"] = _now_iso()
            payload = deepcopy(task)
        else:
            payload = get_task_payload("auto_train", task_id)
            if payload is None:
                return
            payload.update(updates)
            payload["updated_at"] = _now_iso()
    upsert_task_payload("auto_train", payload)


def _snapshot_task(task_id: str) -> AutoTrainTaskResponse | None:
    with AUTO_TRAIN_LOCK:
        task = AUTO_TRAIN_TASKS.get(task_id)
        payload = deepcopy(task) if task is not None else None
    if payload is None:
        payload = get_task_payload("auto_train", task_id)
        if payload is None:
            return None
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
        if active_task["status"] == "stopping":
            raise ValueError("Previous search task is still stopping. Wait a moment and try again.")
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


def _normalize_auto_train_history_summary(task: dict) -> str | None:
    """Build one compact history summary for an auto-train task."""
    summary_payload = task.get("summary") if isinstance(task.get("summary"), dict) else {}
    stop_reason = task.get("stop_reason")
    if isinstance(stop_reason, str) and stop_reason.strip():
        return stop_reason
    error_message = task.get("error")
    if isinstance(error_message, str) and error_message.strip():
        return error_message

    final_proposal = summary_payload.get("final_proposal") if isinstance(summary_payload, dict) else None
    if isinstance(final_proposal, dict):
        final_hypothesis = final_proposal.get("hypothesis")
        if isinstance(final_hypothesis, str) and final_hypothesis.strip():
            return final_hypothesis.strip()

    source_task_type = task.get("source_task_type")
    source_model_name = task.get("source_model_name")
    if source_task_type == "model_compare" and isinstance(source_model_name, str) and source_model_name.strip():
        return f"Search from compare · {source_model_name.strip()}"

    current_round = task.get("current_round")
    if isinstance(current_round, int) and current_round > 0:
        return f"Round {current_round} in progress."
    return None


def _build_search_scope_summary(config: ExperimentConfig) -> str:
    """Build one compact search-scope label from the current search policy."""
    policy = config.search_policy
    categories: list[str] = []
    if policy.allow_basic_hparam_search:
        categories.append("Basic")
    if policy.allow_loss_search:
        categories.append("Loss")
    if policy.allow_augmentation_search:
        categories.append("Data Augmentation")
    if policy.allow_model_module_search:
        categories.append("Architecture")
    return " / ".join(categories) if categories else "No search scope selected"


def _is_retryable_proposal_error(error: Exception) -> bool:
    """Return whether one proposal error is worth retrying after a short delay."""
    if isinstance(error, requests.RequestException):
        return True
    if isinstance(error, AIHubMixRequestError):
        error_text = str(error)
        if any(f"status={status_code}" in error_text for status_code in ("429", "500", "502", "503", "504", "529")):
            return True
        return "overloaded_error" in error_text or "high load" in error_text.lower()
    return False


def _sleep_with_stop_check(task_id: str, seconds: int) -> None:
    """Sleep in short intervals so user stop requests can interrupt retry waits."""
    for _ in range(max(0, seconds)):
        task = _snapshot_task(task_id)
        if task is None or task.stop_requested:
            raise AutoTrainStoppedError("Auto train stopped by user request")
        time.sleep(1)


def _generate_auto_train_proposal(
    task_id: str,
    run_id: str,
    *,
    require_non_basic_change: bool,
) -> object:
    """Generate one proposal with retry-on-network behavior for transient provider failures."""
    total_attempts = len(AUTO_TRAIN_PROPOSAL_RETRY_DELAYS_SECONDS) + 1
    for attempt_index in range(total_attempts):
        db = SessionLocal()
        try:
            return generate_aihubmix_proposal(
                db,
                run_id,
                require_non_basic_change=require_non_basic_change,
                on_prompt_metadata=lambda prompt_metadata: _record_prompt_token_estimate(
                    task_id,
                    prompt_metadata,
                ),
                on_provider_metadata=lambda provider_metadata: _record_provider_usage(
                    task_id,
                    provider_metadata,
                ),
            )
        except ValueError:
            raise
        except Exception as error:
            if not _is_retryable_proposal_error(error) or attempt_index >= total_attempts - 1:
                raise
            retry_delay_seconds = AUTO_TRAIN_PROPOSAL_RETRY_DELAYS_SECONDS[attempt_index]
            _append_task_log(
                task_id,
                (
                    f"Proposal request failed ({attempt_index + 1}/{total_attempts}) | "
                    f"{error} | retrying in {retry_delay_seconds}s"
                ),
            )
            _sleep_with_stop_check(task_id, retry_delay_seconds)
        finally:
            db.close()
    raise RuntimeError("Proposal retry loop exited unexpectedly")


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


def _resolve_structured_train_hyp_changes(proposal_payload: dict) -> dict:
    """Return the structured train_hyp change payload for one proposal."""
    if isinstance(proposal_payload.get("train_hyp_changes"), dict):
        return proposal_payload["train_hyp_changes"]
    return build_train_hyp_change_payload(proposal_payload.get("changes") or {})


def _resolve_structured_recipe_changes(proposal_payload: dict) -> dict:
    """Return the structured model_recipe change payload for one proposal."""
    if isinstance(proposal_payload.get("recipe_changes"), dict):
        return proposal_payload["recipe_changes"]
    return build_model_recipe_change_payload(proposal_payload.get("changes") or {})


def _build_followup_config(latest_experiment: dict, proposal_payload: dict) -> dict:
    latest_config = ExperimentConfig.model_validate(latest_experiment["config"])
    train_hyp_changes = _resolve_structured_train_hyp_changes(proposal_payload)
    recipe_changes = _resolve_structured_recipe_changes(proposal_payload)
    if train_hyp_changes:
        updated_train_hyp = apply_train_hyp_change_payload(
            latest_config.train_hyp.model_dump(),
            train_hyp_changes,
        )
    else:
        updated_train_hyp = apply_proposal_changes_to_train_hyp(
            latest_config.train_hyp.model_dump(),
            proposal_payload.get("changes") or {},
        )
    if recipe_changes:
        updated_model_recipe = apply_model_recipe_change_payload(
            latest_config.model_recipe.model_dump(),
            recipe_changes,
        )
    else:
        updated_model_recipe = apply_proposal_changes_to_model_recipe(
            latest_config.model_recipe.model_dump(),
            proposal_payload.get("changes") or {},
        )
    updated_config = latest_config.model_copy(
        update={
            "train_hyp": updated_train_hyp,
            "model_recipe": updated_model_recipe,
            "params": updated_train_hyp.to_experiment_params(
                aux_logits=bool(updated_model_recipe.modules.get("aux_logits", False)),
            ),
        }
    )
    return {
        "task_type": updated_config.task_type,
        "dataset": updated_config.dataset,
        "model_family": updated_config.model_family,
        "model_name": updated_config.model_name,
        "parameter_space_version": updated_config.parameter_space_version,
        "use_demo_mode": updated_config.use_demo_mode,
        "participates_in_ranking": updated_config.participates_in_ranking,
        "search_policy": updated_config.search_policy.model_dump(),
        "ranking_policy": updated_config.ranking_policy.model_dump(),
        "params": updated_config.params.model_dump(),
        "model_recipe": updated_config.model_recipe.model_dump(),
        "train_hyp": updated_config.train_hyp.model_dump(),
        "dataset_recipe": updated_config.dataset_recipe.model_dump(),
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


def _load_auto_train_seed_experiment(db: SessionLocal, run_id: str) -> dict | None:
    """Return the persisted successful experiment that model search should continue from."""
    try:
        source_experiment_detail = _resolve_followup_source_experiment(
            db,
            run_id,
            {"based_on_experiment_ids": []},
        )
    except ValueError:
        return None
    if source_experiment_detail["status"] != "success":
        return None
    return source_experiment_detail


def _try_attach_final_proposal(
    task_id: str,
    *,
    run_id: str,
    round_index: int,
    summary: dict | None,
    policy_stop_reason: str | None = None,
) -> None:
    """Attach one next-step proposal to the task summary when auto-train stops."""
    if summary is None:
        return

    db = SessionLocal()
    try:
        proposal = generate_aihubmix_proposal(
            db,
            run_id,
            require_non_basic_change=require_non_basic_change_after_warmup_rounds(round_index),
        )
    except Exception as error:
        fallback_summary = deepcopy(summary)
        fallback_summary["final_proposal"] = None
        fallback_summary["final_suggestion_error"] = str(error)
        if policy_stop_reason:
            fallback_summary["stop_reason"] = policy_stop_reason
        _update_task(task_id, summary=fallback_summary)
        _append_task_log(task_id, f"Final suggestion generation failed: {error}")
    else:
        updated_summary = deepcopy(summary)
        updated_summary["final_proposal"] = proposal.model_dump()
        if policy_stop_reason:
            updated_summary["stop_reason"] = policy_stop_reason
        _update_task(task_id, summary=updated_summary)
        _append_task_log(task_id, "Final next-step suggestion generated")
    finally:
        db.close()


def _run_auto_train_task(task_id: str, request: AutoTrainStartRequest) -> None:
    try:
        task_snapshot = _snapshot_task(task_id)
        if task_snapshot is None or task_snapshot.stop_requested:
            raise AutoTrainStoppedError("Auto train stopped before execution started")
        _update_task(task_id, status="running", activity_message="Validating dataset manifests and training config")
        started_at_monotonic = time.monotonic()
        run_policy = get_default_run_policy()
        db = SessionLocal()
        try:
            if request.run_id:
                _set_activity_message(task_id, "Loading the selected run")
                run_detail = get_run_detail(db, request.run_id)
                if run_detail is None:
                    raise ValueError("Selected run not found")
                run_id = run_detail.id
            else:
                _set_activity_message(task_id, "Creating a new search run")
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
            if not request.run_id:
                _append_owned_run(task_id, run_id)
        finally:
            db.close()

        if request.run_id:
            db = SessionLocal()
            try:
                _set_activity_message(task_id, "Checking for an existing successful baseline")
                baseline_detail = _load_auto_train_seed_experiment(db, run_id)
            finally:
                db.close()
            if baseline_detail is not None:
                _update_task(task_id, current_experiment_id=baseline_detail["id"], current_round=0)
                _append_task_log(task_id, f"Reusing existing baseline: {baseline_detail['id']}")
            else:
                _append_task_log(task_id, f"No successful baseline found in run {run_id}; creating a new baseline")
                _set_activity_message(task_id, "Creating baseline experiment")
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
                _set_activity_message(task_id, "Starting baseline training")
                start_experiment_training(baseline_experiment.id)
                _append_task_log(task_id, f"Baseline training started: {baseline_experiment.id}")
                _set_activity_message(task_id, "Baseline training is running")
                baseline_detail = _wait_for_experiment_terminal(
                    task_id,
                    baseline_experiment.id,
                    started_at_monotonic=started_at_monotonic,
                )
                if baseline_detail["status"] == "discarded":
                    raise AutoTrainStoppedError("Baseline experiment was discarded")
                if baseline_detail["status"] != "success":
                    raise ValueError(f"Baseline experiment failed with status {baseline_detail['status']}")
        else:
            _set_activity_message(task_id, "Creating baseline experiment")
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
            _set_activity_message(task_id, "Starting baseline training")
            start_experiment_training(baseline_experiment.id)
            _append_task_log(task_id, f"Baseline training started: {baseline_experiment.id}")
            _set_activity_message(task_id, "Baseline training is running")
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
            "current_proposal": None,
        }
        _update_task(task_id, summary=summary)
        _append_task_log(task_id, f"Baseline finished: {summary['baseline']['summary']}")

        round_index = 0
        consecutive_invalid_proposals = 0
        while True:
            _update_elapsed_seconds(task_id, started_at_monotonic)

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
                _try_attach_final_proposal(
                    task_id,
                    run_id=run_id,
                    round_index=round_index + 1,
                    summary=summary,
                    policy_stop_reason=stop_reason,
                )
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
            require_non_basic_change = require_non_basic_change_after_warmup_rounds(round_index, policy=run_policy)
            if require_non_basic_change:
                _append_task_log(
                    task_id,
                    f"Round {round_index}: warmup is complete, prioritize augmentation/loss/strategy changes",
                )
            _append_task_log(task_id, f"Round {round_index}: generating AI proposal")
            _set_activity_message(task_id, f"Generating proposal for round {round_index}")

            try:
                proposal = _generate_auto_train_proposal(
                    task_id,
                    run_id,
                    require_non_basic_change=require_non_basic_change,
                )
            except ValueError as error:
                consecutive_invalid_proposals += 1
                _append_task_log(
                    task_id,
                    f"Round {round_index}: invalid proposal ({consecutive_invalid_proposals} consecutive) | {error}",
                )
                if consecutive_invalid_proposals >= AUTO_TRAIN_MAX_CONSECUTIVE_INVALID_PROPOSALS:
                    raise ValueError(
                        "Auto train stopped after repeated invalid proposals. "
                        f"Last error: {error}"
                    ) from error
                _append_task_log(
                    task_id,
                    (
                        f"Round {round_index}: waiting "
                        f"{AUTO_TRAIN_INVALID_PROPOSAL_RETRY_DELAY_SECONDS}s before retrying proposal generation"
                    ),
                )
                _sleep_with_stop_check(task_id, AUTO_TRAIN_INVALID_PROPOSAL_RETRY_DELAY_SECONDS)
                continue
            consecutive_invalid_proposals = 0

            proposal_payload = proposal.model_dump()
            summary["current_proposal"] = proposal_payload
            _update_task(task_id, summary=summary)
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
            _set_activity_message(task_id, f"Creating experiment for round {round_index}")

            followup_config = _build_followup_config(source_experiment_detail, proposal_payload)
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
            _set_activity_message(task_id, f"Starting training for round {round_index}")
            start_experiment_training(next_experiment.id)
            _append_task_log(task_id, f"Round {round_index}: training started for {next_experiment.id}")
            _set_activity_message(task_id, f"Training round {round_index}")
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
            summary["current_proposal"] = proposal_payload
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
        task_snapshot = _snapshot_task(task_id)
        current_summary = task_snapshot.summary if task_snapshot is not None else None
        current_run_id = task_snapshot.run_id if task_snapshot is not None else None
        if current_run_id:
            _try_attach_final_proposal(
                task_id,
                run_id=current_run_id,
                round_index=round_index + 1,
                summary=current_summary.model_dump() if hasattr(current_summary, "model_dump") else current_summary,
            )
        _update_task(task_id, status="stopped", current_experiment_id=None, stop_reason="Stopped by user request.")
        _append_task_log(task_id, "Auto train stopped and current experiment discarded")
    except Exception as error:
        _update_task(task_id, status="failed", error=str(error), current_experiment_id=None, activity_message=str(error))
        _append_task_log(task_id, f"Auto train failed: {error}")
        error_detail = getattr(error, "response", None)
        if error_detail is not None and getattr(error_detail, "text", None):
            _append_task_log(task_id, f"Provider detail: {error_detail.text}")


def start_auto_train_task(request: AutoTrainStartRequest) -> AutoTrainTaskResponse:
    """Start one background auto-train task."""
    _ensure_no_active_task()
    task_id = f"auto_{uuid4().hex[:8]}"
    created_at = _now_iso()
    ai_model_name = get_settings().aihubmix_model
    training_image_size = request.config.train_hyp.image_size if request.config.train_hyp is not None else request.config.params.image_size
    dataset_summary = build_dataset_summary_text(
        get_local_dataset_summary(request.dataset),
        training_image_size=training_image_size,
    )
    task_payload = {
        "task_id": task_id,
        "title": request.title or request.run_name,
        "status": "queued",
        "dataset": request.dataset,
        "model_name": request.model_name,
        "policy_preset": request.policy_preset,
        "search_scope_summary": _build_search_scope_summary(request.config),
        "run_id": request.run_id,
        "source_task_type": request.source_task_type,
        "source_task_id": request.source_task_id,
        "source_task_title": request.source_task_title,
        "source_model_name": request.source_model_name,
        "current_round": 0,
        "elapsed_seconds": 0.0,
        "current_experiment_id": None,
        "created_at": created_at,
        "updated_at": created_at,
        "activity_message": "Queued and waiting to validate the dataset",
        "dataset_summary": dataset_summary,
        "training_image_size": training_image_size,
        "ai_model_name": ai_model_name,
        "logs": [],
        "summary": None,
        "error": None,
        "stop_requested": False,
        "stop_reason": None,
        "latest_prompt_tokens_estimate": None,
        "estimated_prompt_tokens_total": 0,
        "latest_prompt_history_items": None,
        "latest_provider_prompt_tokens": None,
        "latest_provider_completion_tokens": None,
        "latest_provider_total_tokens": None,
        "provider_prompt_tokens_total": 0,
        "provider_completion_tokens_total": 0,
        "provider_total_tokens_total": 0,
        "owned_run_ids": [],
    }
    with AUTO_TRAIN_LOCK:
        AUTO_TRAIN_TASKS[task_id] = task_payload
    upsert_task_payload("auto_train", task_payload)
    AUTO_TRAIN_EXECUTOR.submit(_run_auto_train_task, task_id, request)
    return AutoTrainTaskResponse.model_validate(task_payload)


def get_auto_train_task(task_id: str) -> AutoTrainTaskResponse | None:
    """Return one auto-train task snapshot."""
    return _snapshot_task(task_id)


def get_active_auto_train_task() -> AutoTrainTaskResponse | None:
    """Return the current active auto-train task when one is running."""
    with AUTO_TRAIN_LOCK:
        active_task = next(
            (
                task
                for task in AUTO_TRAIN_TASKS.values()
                if task["status"] in {"queued", "running", "stopping"}
            ),
            None,
        )
        payload = deepcopy(active_task) if active_task is not None else None
    if payload is None:
        payload = get_active_task_payload("auto_train")
        if payload is None:
            return None
    return AutoTrainTaskResponse.model_validate(payload)


def list_auto_train_tasks() -> list[TaskHistoryItemResponse]:
    """Return compact auto-train task history items sorted by recency."""
    history_items: list[TaskHistoryItemResponse] = []
    for task in list_task_payloads("auto_train"):
        title = task.get("title") or task.get("model_name") or task["task_id"]
        summary = _normalize_auto_train_history_summary(task)
        history_items.append(
            TaskHistoryItemResponse(
                task_id=task["task_id"],
                task_type="auto_train",
                title=title,
                status=task["status"],
                summary=summary,
                dataset=task.get("dataset"),
                model_name=task.get("model_name"),
                policy_preset=task.get("policy_preset"),
                run_id=task.get("run_id"),
                source_task_type=task.get("source_task_type"),
                source_task_id=task.get("source_task_id"),
                source_task_title=task.get("source_task_title"),
                source_model_name=task.get("source_model_name"),
                created_at=task.get("created_at"),
                updated_at=task.get("updated_at"),
            )
        )
    return history_items


def update_auto_train_task_title(task_id: str, title: str) -> AutoTrainTaskResponse | None:
    """Update one auto-train task title."""
    normalized_title = title.strip()
    if not normalized_title:
        return None
    _update_task(task_id, title=normalized_title)
    return _snapshot_task(task_id)


def stop_auto_train_task(task_id: str) -> AutoTrainTaskResponse | None:
    """Request stop for one running auto-train task."""
    with AUTO_TRAIN_LOCK:
        task = AUTO_TRAIN_TASKS.get(task_id)
        if task is not None:
            task["stop_requested"] = True
            if task["status"] == "queued":
                task["status"] = "stopped"
                task["current_experiment_id"] = None
                task["stop_reason"] = "Stopped by user request."
            elif task["status"] in {"running", "stopping"}:
                task["status"] = "stopping"
            payload = deepcopy(task)
            current_experiment_id = task.get("current_experiment_id")
        else:
            payload = get_task_payload("auto_train", task_id)
            if payload is None:
                return None
            payload["stop_requested"] = True
            if payload.get("status") == "queued":
                payload["status"] = "stopped"
                payload["current_experiment_id"] = None
                payload["stop_reason"] = "Stopped by user request."
            elif payload.get("status") in {"running", "stopping"}:
                payload["status"] = "stopping"
            current_experiment_id = None
    upsert_task_payload("auto_train", payload)
    if current_experiment_id:
        try:
            stop_experiment_training(current_experiment_id)
        except ValueError:
            pass
    return _snapshot_task(task_id)


def delete_auto_train_task(task_id: str) -> dict[str, int] | None:
    """Delete one stopped auto-train task and its task-owned runs."""
    task_snapshot = _snapshot_task(task_id)
    if task_snapshot is None:
        return None
    if task_snapshot.status in {"queued", "running", "stopping"}:
        raise ValueError("Active search tasks must be stopped before deletion.")

    payload = get_task_payload("auto_train", task_id)
    if payload is None:
        return None

    deleted_runs = 0
    deleted_experiments = 0
    deleted_results = 0
    deleted_artifact_files = 0
    for run_id in list(dict.fromkeys(payload.get("owned_run_ids") or [])):
        db = SessionLocal()
        try:
            deleted_counts = clear_run_records(db, run_id)
        finally:
            db.close()
        if deleted_counts is None:
            continue
        deleted_runs += deleted_counts.get("deleted_runs", 0)
        deleted_experiments += deleted_counts.get("deleted_experiments", 0)
        deleted_results += deleted_counts.get("deleted_results", 0)
        deleted_artifact_files += deleted_counts.get("deleted_artifact_files", 0)

    with AUTO_TRAIN_LOCK:
        AUTO_TRAIN_TASKS.pop(task_id, None)
    delete_task_payload("auto_train", task_id)
    return {
        "deleted_tasks": 1,
        "deleted_runs": deleted_runs,
        "deleted_experiments": deleted_experiments,
        "deleted_results": deleted_results,
        "deleted_artifact_files": deleted_artifact_files,
    }
