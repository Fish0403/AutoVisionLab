"""Background cross-model baseline comparison service."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Lock
import time
from uuid import uuid4

from app.db.session import SessionLocal
from app.schemas.experiment import ExperimentCreateRequest
from app.schemas.parameter_space import ExperimentConfig, SearchPolicy
from app.schemas.run import (
    ModelCompareCandidateResult,
    ModelCompareStartRequest,
    ModelCompareSummary,
    ModelCompareTaskResponse,
    RunCreateRequest,
)
from app.services.auto_train_service import get_active_auto_train_task
from app.services.parameter_space import get_parameter_space
from app.services.persistence import create_experiment, create_run, get_experiment_detail
from app.services.training_runner import start_experiment_training


MODEL_COMPARE_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="autovisionlab-model-compare")
MODEL_COMPARE_TASKS: dict[str, dict] = {}
MODEL_COMPARE_LOCK = Lock()
DEFAULT_COMPARE_CANDIDATE_MODELS = ("mobilenet_v2", "mobilenet_v3_small", "googlenet")


def _append_task_log(task_id: str, message: str) -> None:
    with MODEL_COMPARE_LOCK:
        task = MODEL_COMPARE_TASKS.get(task_id)
        if task is None:
            return
        task["logs"].append(message)
        task["logs"] = task["logs"][-200:]


def _update_task(task_id: str, **updates) -> None:
    with MODEL_COMPARE_LOCK:
        task = MODEL_COMPARE_TASKS.get(task_id)
        if task is None:
            return
        task.update(updates)


def _snapshot_task(task_id: str) -> ModelCompareTaskResponse | None:
    with MODEL_COMPARE_LOCK:
        task = MODEL_COMPARE_TASKS.get(task_id)
        if task is None:
            return None
        payload = deepcopy(task)
    return ModelCompareTaskResponse.model_validate(payload)


def _ensure_no_active_compare_task() -> None:
    with MODEL_COMPARE_LOCK:
        active_task = next(
            (
                task
                for task in MODEL_COMPARE_TASKS.values()
                if task["status"] in {"queued", "running"}
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
            "augmentation_policy": params.augmentation_policy,
            "augmentation_params": params.augmentation_params.model_dump(),
            "loss_name": params.loss_name,
            "loss_params": params.loss_params.model_dump(),
            "label_smoothing": params.label_smoothing,
        },
    }


def _build_compare_config(
    *,
    base_config: ExperimentConfig,
    model_name: str,
) -> tuple[ExperimentConfig, list[str]]:
    """Build one normalized experiment config for fair cross-model comparison."""
    parameter_space = get_parameter_space(model_name)
    if parameter_space is None:
        raise ValueError(f"Parameter space not found for compare model {model_name}")

    params_payload = base_config.params.model_dump()
    notes: list[str] = []
    if model_name == "googlenet":
        if params_payload.get("aux_logits") not in {None, False}:
            notes.append("Forced aux_logits=False for fair cross-model comparison.")
        params_payload["aux_logits"] = False
    else:
        params_payload["aux_logits"] = None
    if model_name == "mobilenet_v3_small":
        notes.append("Disabled component search and used the default native recipe.")

    compare_payload = {
        "task_type": base_config.task_type,
        "dataset": base_config.dataset,
        "model_family": _infer_model_family(model_name),
        "model_name": model_name,
        "parameter_space_version": parameter_space.version,
        "use_demo_mode": base_config.use_demo_mode,
        "participates_in_ranking": base_config.participates_in_ranking,
        "search_policy": _build_shared_search_policy().model_dump(),
        "ranking_policy": base_config.ranking_policy.model_dump(),
        "params": params_payload,
    }
    return ExperimentConfig.model_validate(compare_payload), notes


def _build_compare_run_name(dataset: str, model_name: str, task_id: str) -> str:
    """Build one readable run name for a compare candidate."""
    normalized_dataset = dataset.replace(" ", "-").lower()
    normalized_model = model_name.replace("_", "-")
    return f"{normalized_dataset}-{normalized_model}-compare-{task_id[:4]}"


def _wait_for_experiment_terminal(task_id: str, experiment_id: str, *, started_at_monotonic: float) -> dict:
    """Poll experiment status until one terminal result is available."""
    while True:
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
    )
    successful_candidate_count = 0

    try:
        for model_index, model_name in enumerate(candidate_models, start=1):
            _update_task(
                task_id,
                current_model_name=model_name,
                current_model_index=model_index,
                current_run_id=None,
                current_experiment_id=None,
            )
            _append_task_log(task_id, f"[{model_index}/{len(candidate_models)}] Comparing {model_name}")

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

                started_experiment = start_experiment_training(experiment_id)
                if started_experiment is None:
                    raise ValueError("Failed to start compare experiment")
                _append_task_log(task_id, f"{model_name}: started baseline experiment {experiment_id}")

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
        _update_task(
            task_id,
            status=final_status,
            current_model_name=None,
            current_run_id=None,
            current_experiment_id=None,
            elapsed_seconds=max(0.0, time.monotonic() - started_at_monotonic),
            summary=summary.model_dump(),
            error=final_error,
        )
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
    task_payload = {
        "task_id": task_id,
        "status": "queued",
        "elapsed_seconds": 0.0,
        "current_model_name": None,
        "current_model_index": 0,
        "total_models": len(candidate_models),
        "current_run_id": None,
        "current_experiment_id": None,
        "logs": ["Model compare task queued."],
        "summary": initial_summary.model_dump(),
        "error": None,
    }
    with MODEL_COMPARE_LOCK:
        MODEL_COMPARE_TASKS[task_id] = task_payload
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
                if task["status"] in {"queued", "running"}
            ),
            None,
        )
    if active_task_id is None:
        return None
    return _snapshot_task(active_task_id)
