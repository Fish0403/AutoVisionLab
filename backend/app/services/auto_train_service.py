"""Background auto-train orchestration for the demo backend."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
from threading import Lock
import time
from typing import Callable
from uuid import uuid4

import requests

from app.core.settings import get_settings
from app.db.session import SessionLocal
from app.llm.aihubmix_client import AIHubMixClient, AIHubMixRequestError
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
from app.services.dataset_service import (
    build_dataset_summary_text,
    build_lightweight_dataset_summary_text,
    get_lightweight_local_dataset_summary,
    get_local_dataset_summary,
)
from app.services.parameter_space import is_epoch_search_enabled
from app.services.persistence import (
    clear_run_records,
    create_experiment,
    create_run,
    discard_experiment,
    get_experiment_detail,
    get_run_detail,
)
from app.services.proposal_service import generate_aihubmix_proposal, get_run_history_payload
from app.services.task_store import delete_task_payload, get_active_task_payload, get_task_payload, list_task_payloads, upsert_task_payload
from app.services.training_runner import start_experiment_training, stop_experiment_training
from app.trainers.builder import build_model_from_config


AUTO_TRAIN_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="autovisionlab-auto-train")
AUTO_TRAIN_TASKS: dict[str, dict] = {}
AUTO_TRAIN_LOCK = Lock()
AUTO_TRAIN_EXPERIMENT_RETRY_BASE_SECONDS = 5
AUTO_TRAIN_EXPERIMENT_RETRY_CAP_SECONDS = 60
AUTO_TRAIN_MAX_BASELINE_ATTEMPTS = 3
AUTO_TRAIN_MAX_EXPERIMENT_ATTEMPTS_PER_ROUND = 3
AUTO_TRAIN_MIN_ROUNDS_FOR_AI_SUMMARY = 3
AUTO_TRAIN_PROPOSAL_RETRY_SCHEDULE_SECONDS = (30, 60, 180)


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
    resource = (experiment_detail.get("result") or {}).get("resource") or {}
    summary_parts = []
    for key in ("top1_acc", "val_loss", "train_loss", "best_epoch"):
        value = metrics.get(key)
        if value is not None:
            summary_parts.append(f"{key}={value}")
    if summary_parts:
        summary_text = ", ".join(summary_parts)
    else:
        summary_text = (
            experiment_detail.get("decision_reason")
            or experiment_detail.get("status")
            or "no metrics returned"
        )
    return {
        "experiment_id": experiment_detail["id"],
        "status": experiment_detail["status"],
        "decision": experiment_detail.get("decision"),
        "decision_reason": experiment_detail.get("decision_reason"),
        "metrics": metrics,
        "resource": resource,
        "summary": summary_text,
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

    current_proposal = summary_payload.get("current_proposal") if isinstance(summary_payload, dict) else None
    if isinstance(current_proposal, dict):
        current_hypothesis = current_proposal.get("hypothesis")
        if isinstance(current_hypothesis, str) and current_hypothesis.strip():
            return current_hypothesis.strip()

    rounds = summary_payload.get("rounds") if isinstance(summary_payload, dict) else None
    if isinstance(rounds, list):
        for round_payload in reversed(rounds):
            proposal_payload = round_payload.get("proposal") if isinstance(round_payload, dict) else None
            if isinstance(proposal_payload, dict):
                round_hypothesis = proposal_payload.get("hypothesis")
                if isinstance(round_hypothesis, str) and round_hypothesis.strip():
                    return round_hypothesis.strip()

    source_task_type = task.get("source_task_type")
    source_model_name = task.get("source_model_name")
    if source_task_type == "model_compare" and isinstance(source_model_name, str) and source_model_name.strip():
        return f"Search from compare · {source_model_name.strip()}"

    current_round = task.get("current_round")
    if isinstance(current_round, int) and current_round > 0:
        return f"Round {current_round} in progress."
    return None


def _build_search_scope_summary(search_policy: object) -> str:
    """Build one compact search-scope label from the active search policy."""
    effective_policy = SearchPolicy.model_validate(search_policy or {})
    categories: list[str] = []
    if effective_policy.allow_basic_hparam_search:
        categories.append("Basic")
    if is_epoch_search_enabled(effective_policy):
        categories.append("Training Budget")
    if effective_policy.allow_loss_search:
        categories.append("Loss")
    if effective_policy.allow_augmentation_search:
        categories.append("Data Augmentation")
    if effective_policy.allow_model_module_search:
        categories.append("Architecture")
    return " / ".join(categories) if categories else "All supported settings"


def _is_retryable_proposal_error(error: Exception) -> bool:
    """Return whether one proposal error is worth retrying after a short delay."""
    if isinstance(error, ValueError):
        normalized_error_text = str(error).lower()
        return any(
            marker in normalized_error_text
            for marker in (
                "proposal",
                "json",
                "field",
                "changes",
                "parameter",
                "range",
                "schema",
                "unsupported",
                "model_name",
            )
        )
    if isinstance(error, requests.RequestException):
        return True
    if isinstance(error, AIHubMixRequestError):
        error_text = str(error)
        if any(f"status={status_code}" in error_text for status_code in ("429", "500", "502", "503", "504", "529")):
            return True
        normalized_error_text = error_text.lower()
        return (
            "overloaded_error" in error_text
            or "high load" in normalized_error_text
            or "read timed out" in normalized_error_text
            or "connect timeout" in normalized_error_text
        )
    return False


def _build_capped_retry_delay_seconds(attempt_number: int, *, base_seconds: int, cap_seconds: int) -> int:
    """Return one exponential backoff delay capped to the configured maximum."""
    effective_attempt_number = max(1, attempt_number)
    return min(cap_seconds, base_seconds * (2 ** (effective_attempt_number - 1)))


def _build_proposal_retry_delay_seconds(attempt_number: int) -> int:
    """Return the proposal retry delay from the fixed provider-backoff schedule."""
    effective_attempt_number = max(1, attempt_number)
    schedule_index = min(effective_attempt_number - 1, len(AUTO_TRAIN_PROPOSAL_RETRY_SCHEDULE_SECONDS) - 1)
    return AUTO_TRAIN_PROPOSAL_RETRY_SCHEDULE_SECONDS[schedule_index]


def _sleep_with_stop_check(task_id: str, seconds: int) -> None:
    """Sleep in short intervals so user stop requests can interrupt retry waits."""
    for _ in range(max(0, seconds)):
        task = _snapshot_task(task_id)
        if task is None or task.stop_requested:
            raise AutoTrainStoppedError("Auto train stopped by user request")
        time.sleep(1)


def _build_error_summary_snapshot(
    *,
    experiment_id: str | None,
    status: str,
    summary_text: str,
) -> dict:
    """Build one synthetic summary snapshot when no experiment result payload exists."""
    return {
        "experiment_id": experiment_id,
        "status": status,
        "decision": "crash" if status == "failed" else "discard",
        "decision_reason": summary_text,
        "metrics": {},
        "summary": summary_text,
    }


def _normalize_status_fragment(text: str | None) -> str:
    """Return one short status fragment without trailing sentence punctuation."""
    normalized_text = (text or "").strip()
    return normalized_text.rstrip(".!?:;，。！？：； ").strip()


def _build_auto_train_summary_prompt(
    *,
    run_payload: dict[str, object],
    experiment_history: list[dict[str, object]],
    search_summary: dict[str, object],
    stop_reason: str | None,
) -> tuple[str, str]:
    """Build the prompt pair for one stopped-search summary."""
    system_prompt = (
        "You summarize the outcome of an image classification search task for a machine learning workspace. "
        "Return strict JSON with one key: summary_text. "
        "The summary_text must be factual, written in English, and formatted as exactly four sentences. "
        "Do not mention being an AI. Do not recommend next steps. "
        "Sentence 1 must state why the search ended and the overall search scope. "
        "Sentence 2 must identify the leading experiment, include its experiment id, and summarize its key metrics when available. "
        "Sentence 3 must summarize the main strategy or strategies that produced the strongest gains or the most stable results. "
        "Sentence 4 must summarize the strategy or strategies that were ineffective, unstable, or repeatedly unsuccessful. "
        "If the search stopped by user request, state that neutrally. "
        "If no experiment succeeded, state that clearly and still keep the four-sentence format. "
        "If the history does not support a clear positive or negative trend, say that explicitly."
    )
    user_prompt = (
        "Summarize the following search task for one workspace results panel.\n"
        f"Run summary:\n{json.dumps(run_payload, ensure_ascii=True)}\n"
        f"Task summary:\n{json.dumps(search_summary, ensure_ascii=True)}\n"
        f"Experiment history:\n{json.dumps(experiment_history, ensure_ascii=True)}\n"
        f"Stop reason:\n{json.dumps(stop_reason, ensure_ascii=True)}\n"
    )
    return system_prompt, user_prompt


def _generate_auto_train_ai_summary(
    run_id: str,
    search_summary: dict[str, object],
    *,
    stop_reason: str | None,
) -> str | None:
    """Generate one concise summary for a stopped auto-train task."""
    db = SessionLocal()
    try:
        run_detail = get_run_detail(db, run_id)
        if run_detail is None:
            return None
        experiment_history = get_run_history_payload(db, run_id)
    finally:
        db.close()

    if not experiment_history:
        return None

    run_payload = {
        "id": run_detail.id,
        "name": run_detail.name,
        "dataset": run_detail.dataset,
        "model_name": run_detail.model_name,
        "best_experiment_id": run_detail.best_experiment_id,
        "experiment_count": len(experiment_history),
    }
    system_prompt, user_prompt = _build_auto_train_summary_prompt(
        run_payload=run_payload,
        experiment_history=experiment_history,
        search_summary=search_summary,
        stop_reason=stop_reason,
    )
    client = AIHubMixClient()
    response_payload = client.create_json_completion(system_prompt, user_prompt)
    summary_text = response_payload.get("summary_text")
    if not isinstance(summary_text, str):
        raise ValueError("Auto-train summary provider returned an invalid summary_text")
    normalized_summary = " ".join(summary_text.split())
    return normalized_summary or None


def _try_attach_auto_train_ai_summary(
    task_id: str,
    run_id: str,
    search_summary: dict[str, object],
    *,
    stop_reason: str | None,
) -> dict[str, object]:
    """Attach one AI-generated summary to the auto-train payload when possible."""
    updated_summary = deepcopy(search_summary)
    completed_rounds = updated_summary.get("rounds")
    if not isinstance(completed_rounds, list) or len(completed_rounds) < AUTO_TRAIN_MIN_ROUNDS_FOR_AI_SUMMARY:
        updated_summary["ai_summary"] = None
        updated_summary["ai_summary_error"] = None
        return updated_summary
    try:
        ai_summary = _generate_auto_train_ai_summary(
            run_id,
            updated_summary,
            stop_reason=stop_reason,
        )
    except Exception as error:
        _append_task_log(task_id, f"Search summary generation failed: {error}")
        updated_summary["ai_summary"] = None
        updated_summary["ai_summary_error"] = str(error)
        return updated_summary
    updated_summary["ai_summary_error"] = None
    if ai_summary:
        updated_summary["ai_summary"] = ai_summary
        _append_task_log(task_id, "Search summary generated")
    return updated_summary


def _generate_auto_train_proposal(
    task_id: str,
    run_id: str,
    *,
    proposal_validator: Callable[[object], str | None] | None = None,
) -> object:
    """Generate one proposal and keep retrying transient or invalid responses."""
    attempt_index = 0
    retry_feedback: str | None = None
    while True:
        attempt_index += 1
        db = SessionLocal()
        try:
            proposal = generate_aihubmix_proposal(
                db,
                run_id,
                retry_feedback=retry_feedback,
                on_prompt_metadata=lambda prompt_metadata: _record_prompt_token_estimate(
                    task_id,
                    prompt_metadata,
                ),
                on_provider_metadata=lambda provider_metadata: _record_provider_usage(
                    task_id,
                    provider_metadata,
                ),
            )
            if proposal_validator is not None:
                validation_error = proposal_validator(proposal)
                if validation_error:
                    retry_feedback = validation_error
                    _update_task(task_id, proposal_warning=validation_error)
                    raise ValueError(validation_error)
            _update_task(task_id, proposal_warning=None)
            return proposal
        except AutoTrainStoppedError:
            raise
        except Exception as error:
            if not _is_retryable_proposal_error(error):
                raise
            retry_feedback = str(error)
            _update_task(task_id, proposal_warning=str(error))
            retry_delay_seconds = _build_proposal_retry_delay_seconds(attempt_index)
            _append_task_log(
                task_id,
                (
                    f"Proposal attempt failed (attempt {attempt_index}) | "
                    f"{error} | retrying in {retry_delay_seconds}s"
                ),
            )
            _set_activity_message(task_id, f"Proposal retry in {retry_delay_seconds}s")
            _sleep_with_stop_check(task_id, retry_delay_seconds)
        finally:
            db.close()


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
    effective_changes = proposal_payload.get("changes") or {}
    if isinstance(effective_changes, dict) and any(value is not None for value in effective_changes.values()):
        return build_train_hyp_change_payload(effective_changes)
    if isinstance(proposal_payload.get("train_hyp_changes"), dict):
        return proposal_payload["train_hyp_changes"]
    return {}


def _resolve_structured_recipe_changes(proposal_payload: dict) -> dict:
    """Return the structured model_recipe change payload for one proposal."""
    change_payload = proposal_payload.get("changes") or {}
    if isinstance(change_payload, dict) and any(value is not None for value in change_payload.values()):
        return build_model_recipe_change_payload(change_payload)
    raw_recipe_changes = proposal_payload.get("recipe_changes")
    if isinstance(raw_recipe_changes, dict):
        return raw_recipe_changes
    return {}


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


def _validate_followup_proposal(run_id: str, proposal_payload: dict) -> str | None:
    """Return one rejection reason when a proposal cannot build a valid follow-up config."""
    db = SessionLocal()
    try:
        source_experiment_detail = _resolve_followup_source_experiment(db, run_id, proposal_payload)
    finally:
        db.close()
    try:
        followup_config = _build_followup_config(source_experiment_detail, proposal_payload)
        build_model_from_config(ExperimentConfig.model_validate(followup_config))
    except Exception as error:
        return f"Proposal cannot build a valid follow-up config: {error}"
    return None


def _resolve_followup_source_experiment(db: SessionLocal, run_id: str, proposal_payload: dict) -> dict:
    """Pick the current best experiment that the next round should branch from."""
    run_detail = get_run_detail(db, run_id)
    if run_detail is None:
        raise ValueError("Run not found during auto train")

    if run_detail.best_experiment_id is not None:
        best_experiment_detail = get_experiment_detail(db, run_detail.best_experiment_id)
        if best_experiment_detail is not None:
            return best_experiment_detail.model_dump()

    if run_detail.experiments:
        latest_experiment_detail = get_experiment_detail(db, run_detail.experiments[-1].id)
        if latest_experiment_detail is not None:
            return latest_experiment_detail.model_dump()

    raise ValueError("No valid source experiment found for the next auto-train round")


def _load_auto_train_seed_experiment(db: SessionLocal, run_id: str) -> dict | None:
    """Return the latest persisted seed experiment that model search should continue from."""
    try:
        source_experiment_detail = _resolve_followup_source_experiment(
            db,
            run_id,
            {"based_on_experiment_ids": []},
        )
    except ValueError:
        return None
    return source_experiment_detail


def _maybe_discard_non_terminal_experiment(experiment_id: str) -> dict | None:
    """Best-effort discard for queued or running experiments after one failed attempt."""
    db = SessionLocal()
    try:
        experiment_detail = get_experiment_detail(db, experiment_id)
        if experiment_detail is None:
            return None
        detail_payload = experiment_detail.model_dump()
        if detail_payload["status"] not in {"success", "failed", "discarded"}:
            discarded_detail = discard_experiment(db, experiment_id)
            if discarded_detail is not None:
                return discarded_detail.model_dump()
        return detail_payload
    finally:
        db.close()


def _run_experiment_with_retries(
    task_id: str,
    *,
    label: str,
    experiment_request: ExperimentCreateRequest,
    started_at_monotonic: float,
    max_attempts: int,
) -> dict:
    """Create, train, and retry one experiment attempt budget before giving up on the round."""
    last_failure_payload: dict | None = None
    for attempt_index in range(1, max_attempts + 1):
        task_snapshot = _snapshot_task(task_id)
        if task_snapshot is None or task_snapshot.stop_requested:
            raise AutoTrainStoppedError("Auto train stopped by user request")

        current_experiment_id: str | None = None
        try:
            _set_activity_message(task_id, f"Creating {label.lower()} attempt {attempt_index}")
            db = SessionLocal()
            try:
                next_experiment = create_experiment(db, experiment_request)
            finally:
                db.close()
            if next_experiment is None:
                raise ValueError(f"Failed to create {label.lower()} experiment")

            current_experiment_id = next_experiment.id
            _update_task(task_id, current_experiment_id=current_experiment_id)
            _append_task_log(
                task_id,
                f"{label} attempt {attempt_index}/{max_attempts}: created experiment {current_experiment_id}",
            )

            _set_activity_message(task_id, f"Starting {label.lower()} attempt {attempt_index}")
            start_experiment_training(current_experiment_id)
            _append_task_log(
                task_id,
                f"{label} attempt {attempt_index}/{max_attempts}: training started for {current_experiment_id}",
            )
            _set_activity_message(task_id, f"Training {label.lower()} attempt {attempt_index}")
            experiment_detail = _wait_for_experiment_terminal(
                task_id,
                current_experiment_id,
                started_at_monotonic=started_at_monotonic,
            )
            if experiment_detail["status"] == "discarded":
                task_snapshot = _snapshot_task(task_id)
                if task_snapshot is None or task_snapshot.stop_requested:
                    raise AutoTrainStoppedError("Auto train stopped by user request")
            if experiment_detail["status"] == "success":
                _update_task(task_id, current_experiment_id=None)
                return experiment_detail
            last_failure_payload = experiment_detail
            failed_attempt_summary = _build_summary_snapshot(experiment_detail)
            _append_task_log(
                task_id,
                (
                    f"{label} attempt {attempt_index}/{max_attempts} failed | "
                    f"{failed_attempt_summary['summary']}"
                ),
            )
        except AutoTrainStoppedError:
            raise
        except Exception as error:
            if current_experiment_id:
                last_failure_detail = _maybe_discard_non_terminal_experiment(current_experiment_id)
                if last_failure_detail is not None:
                    last_failure_payload = last_failure_detail
                    failed_attempt_summary = _build_summary_snapshot(last_failure_detail)
                else:
                    last_failure_payload = _build_error_summary_snapshot(
                        experiment_id=current_experiment_id,
                        status="failed",
                        summary_text=str(error),
                    )
            else:
                last_failure_payload = _build_error_summary_snapshot(
                    experiment_id=None,
                    status="failed",
                    summary_text=str(error),
                )
            failed_attempt_summary = (
                last_failure_payload
                if "summary" in last_failure_payload
                else _build_summary_snapshot(last_failure_payload)
            )
            _append_task_log(
                task_id,
                (
                    f"{label} attempt {attempt_index}/{max_attempts} failed | "
                    f"{failed_attempt_summary['summary']}"
                ),
            )
        finally:
            _update_task(task_id, current_experiment_id=None)

        if attempt_index >= max_attempts:
            break

        retry_delay_seconds = _build_capped_retry_delay_seconds(
            attempt_index,
            base_seconds=AUTO_TRAIN_EXPERIMENT_RETRY_BASE_SECONDS,
            cap_seconds=AUTO_TRAIN_EXPERIMENT_RETRY_CAP_SECONDS,
        )
        _append_task_log(
            task_id,
            (
                f"{label} attempt {attempt_index}/{max_attempts} failed | "
                f"retrying in {retry_delay_seconds}s"
            ),
        )
        _set_activity_message(task_id, f"{label} retry in {retry_delay_seconds}s")
        _sleep_with_stop_check(task_id, retry_delay_seconds)

    return last_failure_payload or _build_error_summary_snapshot(
        experiment_id=None,
        status="failed",
        summary_text=f"{label} exhausted its retry budget without a valid result.",
    )


def _run_auto_train_task(task_id: str, request: AutoTrainStartRequest) -> None:
    run_id: str | None = None
    summary: dict[str, object] | None = None
    try:
        task_snapshot = _snapshot_task(task_id)
        if task_snapshot is None or task_snapshot.stop_requested:
            raise AutoTrainStoppedError("Auto train stopped before execution started")
        _update_task(task_id, status="running", activity_message="Validating dataset manifests and training config")
        started_at_monotonic = time.monotonic()
        _update_task(
            task_id,
            dataset_summary=build_dataset_summary_text(
                get_local_dataset_summary(request.dataset),
                training_image_size=request.config.train_hyp.image_size,
            ),
        )
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

        baseline_request = ExperimentCreateRequest(
            run_id=run_id,
            config=request.config,
            parameter_space=request.parameter_space,
            proposal=None,
        )
        if request.run_id:
            db = SessionLocal()
            try:
                _set_activity_message(task_id, "Checking for an existing seed experiment")
                baseline_detail = _load_auto_train_seed_experiment(db, run_id)
            finally:
                db.close()
            if baseline_detail is not None:
                _update_task(task_id, current_experiment_id=baseline_detail["id"], current_round=0)
                _append_task_log(
                    task_id,
                    f"Reusing existing seed experiment: {baseline_detail['id']} ({baseline_detail['status']})",
                )
            else:
                _append_task_log(task_id, f"No existing seed experiment found in run {run_id}; creating a baseline")
                baseline_detail = _run_experiment_with_retries(
                    task_id,
                    label="Baseline",
                    experiment_request=baseline_request,
                    started_at_monotonic=started_at_monotonic,
                    max_attempts=AUTO_TRAIN_MAX_BASELINE_ATTEMPTS,
                )
        else:
            baseline_detail = _run_experiment_with_retries(
                task_id,
                label="Baseline",
                experiment_request=baseline_request,
                started_at_monotonic=started_at_monotonic,
                max_attempts=AUTO_TRAIN_MAX_BASELINE_ATTEMPTS,
            )

        if not (baseline_detail.get("id") or baseline_detail.get("experiment_id")):
            raise ValueError("Auto train could not create any baseline experiment after retries.")

        summary = {
            "mode": "auto",
            "run_id": run_id,
            "baseline": baseline_detail if "summary" in baseline_detail else _build_summary_snapshot(baseline_detail),
            "rounds": [],
            "current_proposal": None,
        }
        _update_task(task_id, summary=summary, proposal_warning=None)
        _append_task_log(task_id, f"Baseline finished: {summary['baseline']['summary']}")
        if baseline_detail["status"] != "success":
            baseline_snapshot = summary["baseline"]
            baseline_experiment_id = baseline_snapshot.get("experiment_id") or "unknown"
            baseline_failure_summary = _normalize_status_fragment(
                baseline_snapshot.get("summary") or baseline_detail.get("status") or "unknown error"
            )
            raise ValueError(
                f"Baseline {baseline_experiment_id} failed: {baseline_failure_summary}. "
                "Auto train stopped before generating any AI proposals."
            )

        round_index = 0
        while True:
            _update_elapsed_seconds(task_id, started_at_monotonic)
            task = _snapshot_task(task_id)
            if task is None or task.stop_requested:
                raise AutoTrainStoppedError("Auto train stopped by user request")
            round_index += 1
            _update_task(task_id, current_round=round_index)
            _append_task_log(task_id, f"Round {round_index}: generating AI proposal")
            _set_activity_message(task_id, f"Generating proposal for round {round_index}")
            _update_task(task_id, proposal_warning=None)
            proposal = _generate_auto_train_proposal(
                task_id,
                run_id,
                proposal_validator=lambda current_proposal: _validate_followup_proposal(
                    run_id,
                    current_proposal.model_dump(),
                ),
            )

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

            followup_config = _build_followup_config(source_experiment_detail, proposal_payload)
            current_experiment_detail = _run_experiment_with_retries(
                task_id,
                label=f"Round {round_index}",
                experiment_request=ExperimentCreateRequest(
                    run_id=run_id,
                    config=followup_config,
                    parameter_space=request.parameter_space,
                    proposal=proposal,
                ),
                started_at_monotonic=started_at_monotonic,
                max_attempts=AUTO_TRAIN_MAX_EXPERIMENT_ATTEMPTS_PER_ROUND,
            )

            summary["rounds"].append(
                {
                    "round_index": round_index,
                    "proposal": proposal_payload,
                    "result": (
                        current_experiment_detail
                        if "summary" in current_experiment_detail
                        else _build_summary_snapshot(current_experiment_detail)
                    ),
                }
            )
            summary["current_proposal"] = proposal_payload
            _update_task(task_id, summary=summary)
            _append_task_log(task_id, f"Round {round_index}: {summary['rounds'][-1]['result']['summary']}")
            if current_experiment_detail.get("status") != "success":
                _append_task_log(
                    task_id,
                    (
                        f"Round {round_index}: retry budget exhausted; "
                        "discarding this round and continuing search"
                    ),
                )
                continue
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
        stop_reason = "Stopped by user request."
        task_snapshot = _snapshot_task(task_id)
        if isinstance(summary, dict):
            summary_payload = deepcopy(summary)
        elif task_snapshot is not None and isinstance(task_snapshot.summary, dict):
            summary_payload = deepcopy(task_snapshot.summary)
        else:
            summary_payload = {
                "mode": "auto",
                "run_id": run_id or (task_snapshot.run_id if task_snapshot is not None else None),
                "baseline": None,
                "rounds": [],
                "current_proposal": None,
            }
        effective_run_id = run_id or (task_snapshot.run_id if task_snapshot is not None else None)
        if effective_run_id and not summary_payload.get("run_id"):
            summary_payload["run_id"] = effective_run_id
        summary_payload["stop_reason"] = stop_reason
        if effective_run_id:
            _set_activity_message(task_id, "Generating search summary")
            summary_payload = _try_attach_auto_train_ai_summary(
                task_id,
                effective_run_id,
                summary_payload,
                stop_reason=stop_reason,
            )
        _update_task(
            task_id,
            status="stopped",
            current_experiment_id=None,
            stop_reason=stop_reason,
            summary=summary_payload,
            proposal_warning=None,
            activity_message=stop_reason,
        )
        _append_task_log(task_id, "Auto train stopped by user request")
    except Exception as error:
        _update_task(
            task_id,
            status="failed",
            error=str(error),
            current_experiment_id=None,
            proposal_warning=None,
            activity_message=str(error),
        )
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
    training_image_size = request.config.train_hyp.image_size
    dataset_summary = build_lightweight_dataset_summary_text(
        get_lightweight_local_dataset_summary(request.dataset),
    )
    task_payload = {
        "task_id": task_id,
        "title": request.title or request.run_name,
        "status": "queued",
        "dataset": request.dataset,
        "model_name": request.model_name,
        "policy_preset": request.policy_preset,
        "search_scope_summary": _build_search_scope_summary(request.config.search_policy),
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
        "proposal_warning": None,
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
