"""Background cross-model baseline comparison service."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import json
from threading import Lock
import time
from uuid import uuid4

from app.core.settings import get_settings
from app.db.session import SessionLocal
from app.llm.aihubmix_client import AIHubMixClient
from app.prompts.compare_summary import build_compare_summary_prompt
from app.schemas.experiment import ExperimentCreateRequest
from app.schemas.parameter_space import ExperimentConfig, SearchPolicy, build_default_model_recipe
from app.schemas.run import (
    ModelCompareCandidateResult,
    ModelCompareStartRequest,
    ModelCompareSummary,
    ModelCompareTaskResponse,
    RunCreateRequest,
    TaskHistoryItemResponse,
)
from app.services.auto_train_service import delete_auto_train_task, get_active_auto_train_task
from app.services.dataset_service import (
    build_dataset_summary_text,
    build_lightweight_dataset_summary_text,
    get_lightweight_local_dataset_summary,
    get_local_dataset_summary,
)
from app.services.parameter_space import get_parameter_space
from app.services.persistence import clear_run_records, create_experiment, create_run, get_experiment_detail
from app.services.task_store import delete_task_payload, get_active_task_payload, get_task_payload, list_task_payloads, upsert_task_payload
from app.services.training_runner import start_experiment_training, stop_experiment_training


MODEL_COMPARE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="autovisionlab-model-compare")
MODEL_COMPARE_TASKS: dict[str, dict] = {}
MODEL_COMPARE_LOCK = Lock()
DEFAULT_COMPARE_CANDIDATE_MODELS = (
    "mobilenet_v2",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "googlenet",
)


class ModelCompareStoppedError(RuntimeError):
    """Raised when one compare task is stopped by user request."""


def _now_iso() -> str:
    """Return one UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def _append_task_log(task_id: str, message: str) -> None:
    with MODEL_COMPARE_LOCK:
        task = MODEL_COMPARE_TASKS.get(task_id)
        if task is None:
            return
        task["logs"].append(message)
        task["logs"] = task["logs"][-200:]
        task["updated_at"] = _now_iso()
        payload = deepcopy(task)
    upsert_task_payload("model_compare", payload)


def _update_task(task_id: str, **updates) -> None:
    with MODEL_COMPARE_LOCK:
        task = MODEL_COMPARE_TASKS.get(task_id)
        if task is not None:
            task.update(updates)
            task["updated_at"] = _now_iso()
            payload = deepcopy(task)
        else:
            payload = get_task_payload("model_compare", task_id)
            if payload is None:
                return
            payload.update(updates)
            payload["updated_at"] = _now_iso()
    upsert_task_payload("model_compare", payload)


def _append_owned_run(task_id: str, run_id: str) -> None:
    """Attach one compare-owned run identifier for later cleanup."""
    with MODEL_COMPARE_LOCK:
        task = MODEL_COMPARE_TASKS.get(task_id)
        if task is not None:
            owned_run_ids = list(task.get("owned_run_ids") or [])
            if run_id not in owned_run_ids:
                owned_run_ids.append(run_id)
            task["owned_run_ids"] = owned_run_ids
            task["updated_at"] = _now_iso()
            payload = deepcopy(task)
        else:
            payload = get_task_payload("model_compare", task_id)
            if payload is None:
                return
            owned_run_ids = list(payload.get("owned_run_ids") or [])
            if run_id not in owned_run_ids:
                owned_run_ids.append(run_id)
            payload["owned_run_ids"] = owned_run_ids
            payload["updated_at"] = _now_iso()
    upsert_task_payload("model_compare", payload)


def _set_activity_message(task_id: str, activity_message: str | None) -> None:
    """Update one short compare activity message for the workspace."""
    _update_task(task_id, activity_message=activity_message)


def _snapshot_task(task_id: str) -> ModelCompareTaskResponse | None:
    with MODEL_COMPARE_LOCK:
        task = MODEL_COMPARE_TASKS.get(task_id)
        payload = deepcopy(task) if task is not None else None
    if payload is None:
        payload = get_task_payload("model_compare", task_id)
        if payload is None:
            return None
    return ModelCompareTaskResponse.model_validate(payload)


def _is_stop_requested(task_id: str) -> bool:
    """Return whether one compare task has a pending stop request."""
    with MODEL_COMPARE_LOCK:
        task = MODEL_COMPARE_TASKS.get(task_id)
        if task is None:
            return False
        return bool(task.get("stop_requested"))


def _ensure_no_active_compare_task() -> None:
    with MODEL_COMPARE_LOCK:
        active_task = next(
            (
                task
                for task in MODEL_COMPARE_TASKS.values()
                if task["status"] in {"queued", "running", "stopping"}
            ),
            None,
        )
    if active_task is not None:
        raise ValueError("Another model compare task is already running")
    active_auto_train_task = get_active_auto_train_task()
    if active_auto_train_task is not None and active_auto_train_task.status in {"queued", "running", "stopping"}:
        raise ValueError("Auto train is already running")


def _infer_model_family(model_name: str) -> str:
    """Infer one model family from its base model name."""
    if model_name.startswith("mobilenet"):
        return "mobilenet"
    if model_name.startswith("efficientnet"):
        return "efficientnet"
    if model_name == "googlenet":
        return "googlenet"
    if model_name.startswith("resnet"):
        return "resnet"
    raise ValueError(f"Unsupported model family for compare task: {model_name}")


def _build_shared_search_policy() -> SearchPolicy:
    """Return one disabled search policy for fair baseline comparison."""
    return SearchPolicy(
        allow_basic_hparam_search=False,
        allowed_basic_hparam_fields=[],
        allow_strategy_search=False,
        allow_loss_search=False,
        allow_augmentation_search=False,
        allow_model_module_search=False,
        require_manual_approval_for_high_impact_changes=True,
    )


def _build_shared_baseline_snapshot(base_config: ExperimentConfig) -> dict[str, object]:
    """Build one compact shared-baseline summary for the compare task."""
    params = base_config.params
    return {
        "dataset": base_config.dataset,
        "use_demo_mode": base_config.use_demo_mode,
        "params": {
            "optimizer": params.optimizer,
            "learning_rate": params.learning_rate,
            "batch_size": params.batch_size,
            "image_size": params.image_size,
            "epochs": params.epochs,
            "weight_decay": params.weight_decay,
            "scheduler": params.scheduler,
            "augmentation_params": params.augmentation_params.model_dump(),
            "loss_name": params.loss_name,
            "loss_params": params.loss_params.model_dump(),
            "label_smoothing": params.label_smoothing,
        },
    }


def _generate_compare_ai_summary(summary: ModelCompareSummary) -> str | None:
    """Generate one concise compare-results summary using AIHubMix."""
    if not summary.candidate_results:
        return None
    if not any(candidate.status == "success" for candidate in summary.candidate_results):
        return None
    system_prompt, user_prompt = _build_compare_summary_prompt(summary)
    client = AIHubMixClient()
    response_payload = client.create_json_completion(system_prompt, user_prompt)
    summary_text = response_payload.get("summary_text")
    if not isinstance(summary_text, str):
        raise ValueError("Model compare summary provider returned an invalid summary_text")
    normalized_summary = " ".join(summary_text.split())
    return normalized_summary or None


def _try_attach_compare_ai_summary(task_id: str, summary: ModelCompareSummary) -> ModelCompareSummary:
    """Attach one AI-generated summary to the compare payload when possible."""
    updated_summary = summary.model_copy(deep=True)
    try:
        ai_summary = _generate_compare_ai_summary(updated_summary)
    except Exception as error:
        _append_task_log(task_id, f"Compare summary generation failed: {error}")
        updated_summary.ai_summary = None
        updated_summary.ai_summary_error = str(error)
        return updated_summary
    updated_summary.ai_summary_error = None
    if ai_summary:
        updated_summary.ai_summary = ai_summary
        _append_task_log(task_id, "Compare summary generated")
    return updated_summary


def _build_compare_config(
    *,
    base_config: ExperimentConfig,
    model_name: str,
) -> tuple[ExperimentConfig, list[str]]:
    """Build one normalized experiment config for fair cross-model comparison."""
    parameter_space = get_parameter_space(model_name)
    if parameter_space is None:
        raise ValueError(f"Parameter space not found for compare model {model_name}")

    compare_model_family = _infer_model_family(model_name)
    compare_model_recipe = build_default_model_recipe(
        model_name=model_name,
        task_type=base_config.task_type,
        model_family=compare_model_family,
    )
    notes: list[str] = []
    if model_name == "googlenet":
        if base_config.use_aux_logits():
            notes.append("Forced aux_logits=False for fair cross-model comparison.")
        compare_model_recipe.modules["aux_logits"] = False
    if model_name in {"mobilenet_v3_small", "mobilenet_v3_large"}:
        notes.append("Disabled component search and used the default native recipe.")

    compare_payload = base_config.model_dump(mode="python")
    compare_payload.update(
        {
            "model_family": compare_model_family,
            "model_name": model_name,
            "parameter_space_version": parameter_space.version,
            "search_policy": _build_shared_search_policy().model_dump(),
            "model_recipe": compare_model_recipe.model_dump(by_alias=True),
        }
    )
    return ExperimentConfig.model_validate(compare_payload), notes


def _build_compare_run_name(dataset: str, model_name: str, task_id: str) -> str:
    """Build one readable run name for a compare candidate."""
    normalized_dataset = dataset.replace(" ", "-").lower()
    normalized_model = model_name.replace("_", "-")
    return f"{normalized_dataset}-{normalized_model}-compare-{task_id[:4]}"


def _wait_for_experiment_terminal(task_id: str, experiment_id: str, *, started_at_monotonic: float) -> dict:
    """Poll experiment status until one terminal result is available."""
    stop_requested = False
    while True:
        if _is_stop_requested(task_id):
            stop_requested = True
        elapsed_seconds = max(0.0, time.monotonic() - started_at_monotonic)
        _update_task(task_id, elapsed_seconds=elapsed_seconds)
        db = SessionLocal()
        try:
            experiment_detail = get_experiment_detail(db, experiment_id)
        finally:
            db.close()
        if experiment_detail is None:
            raise ValueError("Experiment not found during model compare")
        detail_payload = experiment_detail.model_dump()
        if detail_payload["status"] in {"success", "failed", "discarded"}:
            if stop_requested:
                raise ModelCompareStoppedError("Model compare stopped by user request")
            return detail_payload
        time.sleep(1)


def _run_model_compare_task(task_id: str, request: ModelCompareStartRequest) -> None:
    """Run one cross-model baseline comparison in the background."""
    started_at_monotonic = time.monotonic()
    candidate_models = list(request.candidate_models or DEFAULT_COMPARE_CANDIDATE_MODELS)
    base_config = request.config
    summary = ModelCompareSummary(
        shared_baseline_config=_build_shared_baseline_snapshot(base_config),
    )
    _update_task(
        task_id,
        status="running",
        total_models=len(candidate_models),
        summary=summary.model_dump(),
        activity_message="Validating dataset manifests and shared baseline config",
        dataset_summary=build_dataset_summary_text(
            get_local_dataset_summary(request.dataset),
            training_image_size=request.config.train_hyp.image_size,
        ),
    )
    successful_candidate_count = 0

    try:
        for model_index, model_name in enumerate(candidate_models, start=1):
            if _is_stop_requested(task_id):
                raise ModelCompareStoppedError("Model compare stopped by user request")
            _update_task(
                task_id,
                current_model_name=model_name,
                current_model_index=model_index,
                current_run_id=None,
                current_experiment_id=None,
            )
            _append_task_log(task_id, f"[{model_index}/{len(candidate_models)}] Comparing {model_name}")
            _set_activity_message(task_id, f"Preparing shared baseline for {model_name}")

            run_id: str | None = None
            experiment_id: str | None = None
            compare_notes: list[str] = []
            candidate_result = ModelCompareCandidateResult(
                model_name=model_name,
                status="failed",
            )
            try:
                compare_config, compare_notes = _build_compare_config(
                    base_config=base_config,
                    model_name=model_name,
                )
                parameter_space = get_parameter_space(model_name)
                if parameter_space is None:
                    raise ValueError(f"Parameter space not found for compare model {model_name}")

                db = SessionLocal()
                try:
                    _set_activity_message(task_id, f"Creating compare run for {model_name}")
                    run_detail = create_run(
                        db,
                        RunCreateRequest(
                            name=_build_compare_run_name(request.dataset, model_name, task_id),
                            dataset=request.dataset,
                            model_name=model_name,
                            base_config=compare_config,
                            notes="Created by Compare Models.",
                        ),
                    )
                    run_id = run_detail.id
                    _append_owned_run(task_id, run_id)
                    _set_activity_message(task_id, f"Creating baseline experiment for {model_name}")
                    experiment_detail = create_experiment(
                        db,
                        ExperimentCreateRequest(
                            run_id=run_id,
                            config=compare_config,
                            parameter_space=parameter_space,
                            proposal=None,
                        ),
                    )
                    if experiment_detail is None:
                        raise ValueError("Failed to create compare experiment")
                    experiment_id = experiment_detail.id
                finally:
                    db.close()

                candidate_result.run_id = run_id
                candidate_result.baseline_experiment_id = experiment_id
                candidate_result.status = "queued"
                candidate_result.normalized_config_notes = compare_notes
                summary.candidate_results.append(candidate_result)
                _update_task(
                    task_id,
                    current_run_id=run_id,
                    current_experiment_id=experiment_id,
                    summary=summary.model_dump(),
                )

                if _is_stop_requested(task_id):
                    raise ModelCompareStoppedError("Model compare stopped by user request")

                _set_activity_message(task_id, f"Starting baseline training for {model_name}")
                started_experiment = start_experiment_training(experiment_id)
                if started_experiment is None:
                    raise ValueError("Failed to start compare experiment")
                _append_task_log(task_id, f"{model_name}: started baseline experiment {experiment_id}")
                _set_activity_message(task_id, f"Training shared baseline for {model_name}")

                terminal_experiment = _wait_for_experiment_terminal(
                    task_id,
                    experiment_id,
                    started_at_monotonic=started_at_monotonic,
                )
                result_payload = terminal_experiment.get("result") or {}
                metrics_payload = result_payload.get("metrics") or {}
                resource_payload = result_payload.get("resource") or {}
                candidate_result.status = terminal_experiment["status"]
                candidate_result.top1_acc = metrics_payload.get("top1_acc")
                candidate_result.latency_ms = resource_payload.get("latency_ms")
                candidate_result.parameter_count_million = resource_payload.get("parameter_count_million")
                if terminal_experiment["status"] == "success":
                    successful_candidate_count += 1
                _append_task_log(
                    task_id,
                    (
                        f"{model_name}: finished with status {terminal_experiment['status']} | "
                        f"top1_acc={candidate_result.top1_acc} | latency_ms={candidate_result.latency_ms}"
                    ),
                )
            except ModelCompareStoppedError:
                candidate_result.status = "discarded"
                if candidate_result not in summary.candidate_results:
                    candidate_result.run_id = run_id
                    candidate_result.baseline_experiment_id = experiment_id
                    candidate_result.normalized_config_notes = compare_notes
                    summary.candidate_results.append(candidate_result)
                raise
            except Exception as error:
                _append_task_log(task_id, f"{model_name}: compare failed | {error}")
                if candidate_result not in summary.candidate_results:
                    candidate_result.run_id = run_id
                    candidate_result.baseline_experiment_id = experiment_id
                    candidate_result.normalized_config_notes = compare_notes
                    summary.candidate_results.append(candidate_result)
                else:
                    candidate_result.status = "failed"
            finally:
                _update_task(
                    task_id,
                    elapsed_seconds=max(0.0, time.monotonic() - started_at_monotonic),
                    summary=summary.model_dump(),
                )

        final_status = "success" if successful_candidate_count > 0 else "failed"
        final_error = None if successful_candidate_count > 0 else "All model compare candidates failed"
        _set_activity_message(task_id, "Generating compare summary")
        summary = _try_attach_compare_ai_summary(task_id, summary)
        _update_task(
            task_id,
            status=final_status,
            current_model_name=None,
            current_run_id=None,
            current_experiment_id=None,
            elapsed_seconds=max(0.0, time.monotonic() - started_at_monotonic),
            summary=summary.model_dump(),
            error=final_error,
            activity_message=summary.ai_summary or "Compare finished",
        )
    except ModelCompareStoppedError:
        _set_activity_message(task_id, "Generating compare summary")
        summary = _try_attach_compare_ai_summary(task_id, summary)
        _update_task(
            task_id,
            status="stopped",
            current_model_name=None,
            current_run_id=None,
            current_experiment_id=None,
            elapsed_seconds=max(0.0, time.monotonic() - started_at_monotonic),
            summary=summary.model_dump(),
            error=None,
            stop_reason="Stopped by user request.",
            activity_message="Stopped by user request.",
        )
        _append_task_log(task_id, "Model compare stopped and current experiment discarded")
    except Exception as error:
        _update_task(
            task_id,
            status="failed",
            current_model_name=None,
            current_run_id=None,
            current_experiment_id=None,
            elapsed_seconds=max(0.0, time.monotonic() - started_at_monotonic),
            summary=summary.model_dump(),
            error=str(error),
            activity_message=str(error),
        )


def start_model_compare_task(request: ModelCompareStartRequest) -> ModelCompareTaskResponse:
    """Start one cross-model baseline comparison task."""
    _ensure_no_active_compare_task()
    if request.dataset != request.config.dataset:
        raise ValueError("Model compare dataset must match config.dataset")
    task_id = f"cmp_{uuid4().hex[:8]}"
    candidate_models = list(request.candidate_models or DEFAULT_COMPARE_CANDIDATE_MODELS)
    if not candidate_models:
        raise ValueError("Model compare requires at least one candidate model")
    initial_summary = ModelCompareSummary(
        shared_baseline_config=_build_shared_baseline_snapshot(request.config),
    )
    created_at = _now_iso()
    ai_model_name = get_settings().aihubmix_model
    training_image_size = request.config.train_hyp.image_size
    dataset_summary = build_lightweight_dataset_summary_text(
        get_lightweight_local_dataset_summary(request.dataset),
    )
    task_payload = {
        "task_id": task_id,
        "title": request.title or f"Compare Models on {request.dataset}",
        "status": "queued",
        "dataset": request.dataset,
        "candidate_models": candidate_models,
        "elapsed_seconds": 0.0,
        "current_model_name": None,
        "current_model_index": 0,
        "total_models": len(candidate_models),
        "current_run_id": None,
        "current_experiment_id": None,
        "created_at": created_at,
        "updated_at": created_at,
        "activity_message": "Queued and waiting to validate the dataset",
        "dataset_summary": dataset_summary,
        "training_image_size": training_image_size,
        "ai_model_name": ai_model_name,
        "logs": ["Model compare task queued."],
        "summary": initial_summary.model_dump(),
        "error": None,
        "stop_requested": False,
        "stop_reason": None,
        "owned_run_ids": [],
    }
    with MODEL_COMPARE_LOCK:
        MODEL_COMPARE_TASKS[task_id] = task_payload
    upsert_task_payload("model_compare", task_payload)
    MODEL_COMPARE_EXECUTOR.submit(_run_model_compare_task, task_id, request)
    return ModelCompareTaskResponse.model_validate(deepcopy(task_payload))


def get_model_compare_task(task_id: str) -> ModelCompareTaskResponse | None:
    """Get one cross-model compare task snapshot."""
    return _snapshot_task(task_id)


def get_active_model_compare_task() -> ModelCompareTaskResponse | None:
    """Get the currently active cross-model compare task when one exists."""
    with MODEL_COMPARE_LOCK:
        active_task_id = next(
            (
                task_id
                for task_id, task in MODEL_COMPARE_TASKS.items()
                if task["status"] in {"queued", "running", "stopping"}
            ),
            None,
        )
    if active_task_id is not None:
        return _snapshot_task(active_task_id)
    payload = get_active_task_payload("model_compare")
    if payload is None:
        return None
    return ModelCompareTaskResponse.model_validate(payload)


def list_model_compare_tasks() -> list[TaskHistoryItemResponse]:
    """Return compact model-compare task history items sorted by recency."""
    history_items: list[TaskHistoryItemResponse] = []
    for task in list_task_payloads("model_compare"):
        candidate_models = list(task.get("candidate_models") or [])
        summary_payload = task.get("summary") or {}
        ai_summary = summary_payload.get("ai_summary") if isinstance(summary_payload, dict) else None
        summary = ai_summary if isinstance(ai_summary, str) and ai_summary.strip() else None
        stop_reason = task.get("stop_reason")
        current_index = int(task.get("current_model_index") or 0)
        total_models = int(task.get("total_models") or 0)
        if summary:
            pass
        elif stop_reason:
            summary = stop_reason
        elif total_models > 0:
            summary = f"{min(current_index, total_models)}/{total_models} models processed."
        history_items.append(
            TaskHistoryItemResponse(
                task_id=task["task_id"],
                task_type="model_compare",
                title=task.get("title") or f"Compare Models on {task.get('dataset') or 'dataset'}",
                status=task["status"],
                summary=summary,
                dataset=task.get("dataset"),
                candidate_models=candidate_models,
                created_at=task.get("created_at"),
                updated_at=task.get("updated_at"),
            )
        )
    return history_items


def update_model_compare_task_title(task_id: str, title: str) -> ModelCompareTaskResponse | None:
    """Update one model-compare task title."""
    normalized_title = title.strip()
    if not normalized_title:
        return None
    _update_task(task_id, title=normalized_title)
    return _snapshot_task(task_id)


def stop_model_compare_task(task_id: str) -> ModelCompareTaskResponse | None:
    """Request stop for one running compare task."""
    with MODEL_COMPARE_LOCK:
        task = MODEL_COMPARE_TASKS.get(task_id)
        if task is not None:
            task["stop_requested"] = True
            task["stop_reason"] = "Stopped by user request."
            if task["status"] in {"queued", "running"}:
                task["status"] = "stopping"
                task["updated_at"] = _now_iso()
            payload = deepcopy(task)
            current_experiment_id = task.get("current_experiment_id")
        else:
            payload = get_task_payload("model_compare", task_id)
            if payload is None:
                return None
            current_experiment_id = None
    upsert_task_payload("model_compare", payload)
    if current_experiment_id:
        try:
            stop_experiment_training(current_experiment_id)
        except ValueError:
            pass
    return _snapshot_task(task_id)


def delete_model_compare_task(task_id: str) -> dict[str, int] | None:
    """Delete one stopped compare task, linked searches, and compare-owned runs."""
    task_snapshot = _snapshot_task(task_id)
    if task_snapshot is None:
        return None
    if task_snapshot.status in {"queued", "running", "stopping"}:
        raise ValueError("Active compare tasks must be stopped before deletion.")

    payload = get_task_payload("model_compare", task_id)
    if payload is None:
        return None

    linked_search_payloads = [
        search_task
        for search_task in list_task_payloads("auto_train")
        if search_task.get("source_task_type") == "model_compare" and search_task.get("source_task_id") == task_id
    ]
    active_linked_search = next(
        (
            search_task
            for search_task in linked_search_payloads
            if search_task.get("status") in {"queued", "running", "stopping"}
        ),
        None,
    )
    if active_linked_search is not None:
        raise ValueError("Stop linked search tasks before deleting this compare task.")

    deleted_tasks = 1
    deleted_search_tasks = 0
    deleted_runs = 0
    deleted_experiments = 0
    deleted_results = 0
    deleted_artifact_files = 0

    for search_task in linked_search_payloads:
        deleted_counts = delete_auto_train_task(str(search_task["task_id"]))
        if deleted_counts is None:
            continue
        deleted_search_tasks += deleted_counts.get("deleted_tasks", 0)
        deleted_runs += deleted_counts.get("deleted_runs", 0)
        deleted_experiments += deleted_counts.get("deleted_experiments", 0)
        deleted_results += deleted_counts.get("deleted_results", 0)
        deleted_artifact_files += deleted_counts.get("deleted_artifact_files", 0)

    owned_run_ids = list(dict.fromkeys(payload.get("owned_run_ids") or []))
    if not owned_run_ids:
        summary_payload = payload.get("summary") or {}
        candidate_results = summary_payload.get("candidate_results") if isinstance(summary_payload, dict) else []
        for candidate in candidate_results or []:
            run_id = candidate.get("run_id") if isinstance(candidate, dict) else None
            if isinstance(run_id, str) and run_id not in owned_run_ids:
                owned_run_ids.append(run_id)

    for run_id in owned_run_ids:
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

    with MODEL_COMPARE_LOCK:
        MODEL_COMPARE_TASKS.pop(task_id, None)
    delete_task_payload("model_compare", task_id)
    return {
        "deleted_tasks": deleted_tasks,
        "deleted_search_tasks": deleted_search_tasks,
        "deleted_runs": deleted_runs,
        "deleted_experiments": deleted_experiments,
        "deleted_results": deleted_results,
        "deleted_artifact_files": deleted_artifact_files,
    }
