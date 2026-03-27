"""Streamlit demo UI for the autonomous training platform."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import time
from typing import Any

import requests
import streamlit as st


API_BASE_URL = "http://127.0.0.1:8000"
SUPPORTED_MODELS = {
    "mobilenet_v2": {
        "model_family": "mobilenet",
        "parameter_space_version": "mobilenet_v2@v1",
        "parameter_space": {
            "model_name": "mobilenet_v2",
            "version": "mobilenet_v2@v1",
            "editable_params": {
                "optimizer": {"type": "enum", "choices": ["sgd", "adam", "adamw"]},
                "learning_rate": {"type": "number_range", "min": 0.0001, "max": 0.01},
                "batch_size": {"type": "discrete_values", "choices": [32, 64, 128, 256]},
                "image_size": {"type": "discrete_values", "choices": [32, 64, 96]},
                "epochs": {"type": "discrete_values", "choices": [10, 20, 30, 50]},
                "weight_decay": {"type": "number_range", "min": 0.0, "max": 0.01},
                "scheduler": {"type": "enum", "choices": ["none", "step", "cosine"]},
                "augmentation_policy": {"type": "enum", "choices": ["none", "basic"]},
                "mixup_alpha": {"type": "number_range", "min": 0.0, "max": 1.0},
                "cutmix_alpha": {"type": "number_range", "min": 0.0, "max": 1.0},
                "random_erasing_prob": {"type": "number_range", "min": 0.0, "max": 0.5},
                "loss_name": {"type": "enum", "choices": ["cross_entropy", "cross_entropy_with_label_smoothing", "focal_loss"]},
                "focal_gamma": {"type": "number_range", "min": 0.5, "max": 5.0},
                "label_smoothing": {"type": "number_range", "min": 0.0, "max": 0.2},
            },
        },
    },
    "googlenet": {
        "model_family": "googlenet",
        "parameter_space_version": "googlenet@v1",
        "parameter_space": {
            "model_name": "googlenet",
            "version": "googlenet@v1",
            "editable_params": {
                "optimizer": {"type": "enum", "choices": ["sgd", "adam", "adamw"]},
                "learning_rate": {"type": "number_range", "min": 0.0001, "max": 0.01},
                "batch_size": {"type": "discrete_values", "choices": [32, 64, 128, 256]},
                "image_size": {"type": "discrete_values", "choices": [32, 64, 96]},
                "epochs": {"type": "discrete_values", "choices": [10, 20, 30, 50]},
                "weight_decay": {"type": "number_range", "min": 0.0, "max": 0.01},
                "scheduler": {"type": "enum", "choices": ["none", "step", "cosine"]},
                "augmentation_policy": {"type": "enum", "choices": ["none", "basic"]},
                "mixup_alpha": {"type": "number_range", "min": 0.0, "max": 1.0},
                "cutmix_alpha": {"type": "number_range", "min": 0.0, "max": 1.0},
                "random_erasing_prob": {"type": "number_range", "min": 0.0, "max": 0.5},
                "loss_name": {"type": "enum", "choices": ["cross_entropy", "cross_entropy_with_label_smoothing", "focal_loss"]},
                "focal_gamma": {"type": "number_range", "min": 0.5, "max": 5.0},
                "label_smoothing": {"type": "number_range", "min": 0.0, "max": 0.2},
                "aux_logits": {"type": "enum", "choices": [True, False]},
            },
        },
    },
    "resnet18": {
        "model_family": "resnet",
        "parameter_space_version": "resnet18@v1",
        "parameter_space": {
            "model_name": "resnet18",
            "version": "resnet18@v1",
            "editable_params": {
                "optimizer": {"type": "enum", "choices": ["sgd", "adam", "adamw"]},
                "learning_rate": {"type": "number_range", "min": 0.0001, "max": 0.01},
                "batch_size": {"type": "discrete_values", "choices": [32, 64, 128, 256]},
                "image_size": {"type": "discrete_values", "choices": [32, 64, 96]},
                "epochs": {"type": "discrete_values", "choices": [10, 20, 30, 50]},
                "weight_decay": {"type": "number_range", "min": 0.0, "max": 0.01},
                "scheduler": {"type": "enum", "choices": ["none", "step", "cosine"]},
                "augmentation_policy": {"type": "enum", "choices": ["none", "basic"]},
                "mixup_alpha": {"type": "number_range", "min": 0.0, "max": 1.0},
                "cutmix_alpha": {"type": "number_range", "min": 0.0, "max": 1.0},
                "random_erasing_prob": {"type": "number_range", "min": 0.0, "max": 0.5},
                "loss_name": {"type": "enum", "choices": ["cross_entropy", "cross_entropy_with_label_smoothing", "focal_loss"]},
                "focal_gamma": {"type": "number_range", "min": 0.5, "max": 5.0},
                "label_smoothing": {"type": "number_range", "min": 0.0, "max": 0.2},
            },
        },
    },
    "resnet34": {
        "model_family": "resnet",
        "parameter_space_version": "resnet34@v1",
        "parameter_space": {
            "model_name": "resnet34",
            "version": "resnet34@v1",
            "editable_params": {
                "optimizer": {"type": "enum", "choices": ["sgd", "adam", "adamw"]},
                "learning_rate": {"type": "number_range", "min": 0.0001, "max": 0.01},
                "batch_size": {"type": "discrete_values", "choices": [32, 64, 128, 256]},
                "image_size": {"type": "discrete_values", "choices": [32, 64, 96]},
                "epochs": {"type": "discrete_values", "choices": [10, 20, 30, 50]},
                "weight_decay": {"type": "number_range", "min": 0.0, "max": 0.01},
                "scheduler": {"type": "enum", "choices": ["none", "step", "cosine"]},
                "augmentation_policy": {"type": "enum", "choices": ["none", "basic"]},
                "mixup_alpha": {"type": "number_range", "min": 0.0, "max": 1.0},
                "cutmix_alpha": {"type": "number_range", "min": 0.0, "max": 1.0},
                "random_erasing_prob": {"type": "number_range", "min": 0.0, "max": 0.5},
                "loss_name": {"type": "enum", "choices": ["cross_entropy", "cross_entropy_with_label_smoothing", "focal_loss"]},
                "focal_gamma": {"type": "number_range", "min": 0.5, "max": 5.0},
                "label_smoothing": {"type": "number_range", "min": 0.0, "max": 0.2},
            },
        },
    },
    "densenet121": {
        "model_family": "densenet",
        "parameter_space_version": "densenet121@v1",
        "parameter_space": {
            "model_name": "densenet121",
            "version": "densenet121@v1",
            "editable_params": {
                "optimizer": {"type": "enum", "choices": ["sgd", "adam", "adamw"]},
                "learning_rate": {"type": "number_range", "min": 0.0001, "max": 0.01},
                "batch_size": {"type": "discrete_values", "choices": [32, 64, 128, 256]},
                "image_size": {"type": "discrete_values", "choices": [32, 64, 96]},
                "epochs": {"type": "discrete_values", "choices": [10, 20, 30, 50]},
                "weight_decay": {"type": "number_range", "min": 0.0, "max": 0.01},
                "scheduler": {"type": "enum", "choices": ["none", "step", "cosine"]},
                "augmentation_policy": {"type": "enum", "choices": ["none", "basic"]},
                "mixup_alpha": {"type": "number_range", "min": 0.0, "max": 1.0},
                "cutmix_alpha": {"type": "number_range", "min": 0.0, "max": 1.0},
                "random_erasing_prob": {"type": "number_range", "min": 0.0, "max": 0.5},
                "loss_name": {"type": "enum", "choices": ["cross_entropy", "cross_entropy_with_label_smoothing", "focal_loss"]},
                "focal_gamma": {"type": "number_range", "min": 0.5, "max": 5.0},
                "label_smoothing": {"type": "number_range", "min": 0.0, "max": 0.2},
            },
        },
    },
}
MODEL_LABELS = {
    "mobilenet_v2": "MobileNetV2",
    "googlenet": "GoogLeNet",
    "resnet18": "ResNet18",
    "resnet34": "ResNet34",
    "densenet121": "DenseNet121",
}
SUPPORTED_DATASETS = ["cifar10", "neu-cls"]
DATASET_IMAGE_SIZE_OPTIONS = {
    "cifar10": [32, 64, 96],
    "neu-cls": [200, 224, 256],
}
DATASET_RUN_NAME_LABELS = {
    "cifar10": "cifar10",
    "neu-cls": "neu",
}

LOG_LIMIT = 60
LIVE_LOG_CONTAINER: Any | None = None
LIVE_AI_PANEL_CONTAINER: Any | None = None
AI_BLOCKED_CHANGE_FIELDS = {"epochs"}
RANKING_METRIC_LABELS = {
    "top1_acc": "Top1 Acc",
    "val_loss": "Val Loss",
    "training_seconds": "Training Seconds",
}


def default_search_policy() -> dict[str, bool]:
    """Return the default AI search policy for the UI."""
    return {
        "allow_basic_hparam_search": True,
        "allowed_basic_hparam_fields": [
            "optimizer",
            "learning_rate",
            "batch_size",
            "weight_decay",
            "scheduler",
            "label_smoothing",
        ],
        "allow_strategy_search": True,
        "allow_loss_search": True,
        "allow_augmentation_search": True,
        "require_manual_approval_for_high_impact_changes": True,
    }


def default_ranking_policy() -> dict[str, Any]:
    """Return the default ranking policy for the UI."""
    return {
        "primary_metric": "top1_acc",
        "primary_metric_mode": "max",
        "min_primary_metric_improvement": 0.01,
        "primary_metric_parity_epsilon": 0.0005,
        "tie_breaker_metric": "val_loss",
        "tie_breaker_mode": "min",
        "min_tie_breaker_metric_improvement": 0.01,
        "max_image_size": None,
    }


def get_ranking_metric_mode(metric_name: str) -> str:
    """Return the comparison mode for one ranking metric."""
    return "max" if metric_name == "top1_acc" else "min"


def summarize_ranking_policy(ranking_policy: dict[str, Any]) -> str:
    """Build a compact ranking policy summary for the UI."""
    max_image_size = ranking_policy.get("max_image_size")
    max_image_size_label = str(max_image_size) if max_image_size is not None else "none"
    return ", ".join(
        [
            f"primary={ranking_policy.get('primary_metric')}",
            f"min_delta={ranking_policy.get('min_primary_metric_improvement')}",
            f"parity_eps={ranking_policy.get('primary_metric_parity_epsilon')}",
            f"tie_breaker={ranking_policy.get('tie_breaker_metric')}",
            f"tie_delta={ranking_policy.get('min_tie_breaker_metric_improvement')}",
            f"max_image_size={max_image_size_label}",
        ]
    )


def build_auto_run_name(dataset: str, model_name: str) -> str:
    """Build an auto-generated run name from the current dataset and model."""
    dataset_label = DATASET_RUN_NAME_LABELS.get(dataset, dataset)
    model_label = model_name.replace("_", "-")
    date_label = datetime.now().strftime("%Y%m%d")
    return f"{dataset_label}-{model_label}-{date_label}"


def sync_auto_run_name() -> None:
    """Keep the run name aligned with the current dataset and model selection."""
    st.session_state["run_name"] = build_auto_run_name(
        st.session_state["dataset"],
        st.session_state["model_name"],
    )


def get_original_image_size(dataset: str, model_name: str) -> int:
    """Return the dataset-native image size used as the UI default."""
    if dataset == "cifar10":
        return 32
    if dataset == "neu-cls":
        return 200
    return 64


def get_dataset_image_size_options(dataset: str) -> list[int]:
    """Return allowed image size options for one dataset."""
    return DATASET_IMAGE_SIZE_OPTIONS.get(dataset, [get_original_image_size(dataset, "")])


def get_model_label(model_name: str) -> str:
    """Return a user-facing model label."""
    return MODEL_LABELS.get(model_name, model_name)


def get_short_experiment_id(experiment_id: str | None) -> str:
    """Return a short display id for one experiment."""
    if not experiment_id:
        return "-"
    if experiment_id.startswith("exp_"):
        return experiment_id.removeprefix("exp_")[:4]
    return experiment_id[:4]


def format_elapsed_seconds(elapsed_seconds: float) -> str:
    """Format elapsed seconds for one progress caption."""
    total_seconds = max(0, int(elapsed_seconds))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    return f"{minutes}m {seconds}s"


def get_allowed_basic_hparam_fields(dataset: str, model_name: str, image_size: int) -> list[str]:
    """Return the effective basic hyperparameter fields allowed for AI search."""
    fields = [
        "optimizer",
        "learning_rate",
        "batch_size",
        "weight_decay",
        "scheduler",
        "label_smoothing",
    ]
    if image_size != get_original_image_size(dataset, model_name):
        fields.append("image_size")
    return fields


def get_image_size_search_choices(dataset: str, model_name: str, image_size: int) -> list[int]:
    """Return the image_size choices allowed for AI search in the current form state."""
    original_size = get_original_image_size(dataset, model_name)
    lower_bound = min(original_size, image_size)
    upper_bound = max(original_size, image_size)
    all_choices = get_dataset_image_size_options(dataset)
    return [choice for choice in all_choices if lower_bound <= choice <= upper_bound]


def summarize_ai_managed_params(form_values: dict[str, Any]) -> str:
    """Build a compact summary of parameters now managed outside the main form."""
    return ", ".join(
        [
            f"optimizer={form_values['optimizer']}",
            f"weight_decay={form_values['weight_decay']:.4f}",
            f"scheduler={form_values['scheduler']}",
            f"augmentation={form_values['augmentation_policy']}",
            f"label_smoothing={form_values['label_smoothing']:.2f}",
        ]
    )


def summarize_effective_ai_search_fields(
    allow_basic_hparam_search: bool,
    allowed_basic_hparam_fields: list[str],
    allow_strategy_search: bool,
    allow_loss_search: bool,
    allow_augmentation_search: bool,
) -> str:
    """Build a user-facing summary of the current AI search scope."""
    field_labels: list[str] = []
    if allow_basic_hparam_search:
        field_labels.extend(allowed_basic_hparam_fields)
    if allow_strategy_search:
        field_labels.append("aux_logits")
    if allow_loss_search:
        field_labels.extend(["loss_name", "focal_gamma"])
    if allow_augmentation_search:
        field_labels.extend(["augmentation_policy", "mixup_alpha", "cutmix_alpha", "random_erasing_prob"])

    deduped_labels = list(dict.fromkeys(field_labels))
    if not deduped_labels:
        return "当前 AI 不会自动搜索任何参数。"
    return "当前 AI 会搜索： " + ", ".join(deduped_labels)


def request_json(path: str, fallback: Any) -> Any:
    """Fetch JSON from the backend and fall back to local demo data."""
    try:
        response = requests.get(f"{API_BASE_URL}{path}", timeout=0.8)
        response.raise_for_status()
        return unwrap_api_response(response.json())
    except requests.RequestException:
        return fallback


def unwrap_api_response(payload: Any) -> Any:
    """Unwrap the shared API envelope when the backend returns one."""
    if isinstance(payload, dict) and payload.get("ok") is True and "data" in payload:
        return payload["data"]
    return payload


def normalize_error_payload(payload: Any) -> dict[str, Any]:
    """Normalize backend errors into a shape that existing UI code can consume."""
    if isinstance(payload, dict) and payload.get("ok") is False:
        message = payload.get("message")
        if not message:
            first_error = next(
                (
                    error
                    for error in payload.get("errors", [])
                    if isinstance(error, dict) and error.get("message")
                ),
                None,
            )
            message = first_error.get("message") if first_error else None
        normalized_payload = dict(payload)
        normalized_payload["detail"] = message or payload.get("detail") or "Request failed."
        return normalized_payload
    if isinstance(payload, dict):
        return payload if "detail" in payload else {**payload, "detail": str(payload)}
    return {"detail": str(payload)}


def post_json(path: str, payload: dict[str, Any]) -> tuple[bool, Any]:
    """Post JSON to the backend and return success flag with payload or error."""
    try:
        response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=5)
        response.raise_for_status()
        return True, unwrap_api_response(response.json())
    except requests.RequestException as error:
        if getattr(error, "response", None) is not None:
            try:
                return False, normalize_error_payload(error.response.json())
            except ValueError:
                return False, {"detail": error.response.text}
        return False, {"detail": str(error)}


def post_without_body(path: str) -> tuple[bool, Any]:
    """Post without a request body."""
    try:
        response = requests.post(f"{API_BASE_URL}{path}", timeout=3600)
        response.raise_for_status()
        return True, unwrap_api_response(response.json())
    except requests.RequestException as error:
        if getattr(error, "response", None) is not None:
            try:
                return False, normalize_error_payload(error.response.json())
            except ValueError:
                return False, {"detail": error.response.text}
        return False, {"detail": str(error)}


def find_running_experiment_id(runs: list[dict[str, Any]]) -> str | None:
    """Return the currently running experiment id if any."""
    for run in runs:
        run_detail = request_json(f'/runs/{run["id"]}', {})
        for experiment in run_detail.get("experiments", []):
            if experiment.get("status") == "running":
                return experiment.get("id")
    return None


def generate_aihubmix_proposal_request(run_id: str) -> tuple[bool, Any]:
    """Request one real proposal from the backend."""
    try:
        response = requests.post(f"{API_BASE_URL}/runs/{run_id}/proposal", timeout=60)
        response.raise_for_status()
        return True, unwrap_api_response(response.json())
    except requests.RequestException as error:
        if getattr(error, "response", None) is not None:
            try:
                return False, normalize_error_payload(error.response.json())
            except ValueError:
                return False, {"detail": error.response.text}
        return False, {"detail": str(error)}


def start_auto_train_task_request(
    payload: dict[str, Any],
    selected_run_id: str,
    max_wall_clock_minutes: int,
) -> tuple[bool, Any]:
    """Start one backend auto-train task."""
    request_payload = {
        "run_id": None if selected_run_id == "__all__" else selected_run_id,
        "run_name": payload["name"],
        "dataset": payload["dataset"],
        "model_name": payload["model_name"],
        "config": payload["config"],
        "parameter_space": payload["parameter_space"],
        "max_wall_clock_minutes": max_wall_clock_minutes,
    }
    return post_json("/runs/auto-train", request_payload)


def load_auto_train_task(task_id: str) -> tuple[bool, Any]:
    """Load one backend auto-train task."""
    return True, request_json(f"/runs/auto-train/{task_id}", {"detail": "Auto train task not found"})


def stop_auto_train_task_request(task_id: str) -> tuple[bool, Any]:
    """Stop one backend auto-train task."""
    return post_without_body(f"/runs/auto-train/{task_id}/stop")


def load_runs() -> list[dict[str, Any]]:
    """Return all runs."""
    return request_json(
        "/runs",
        [
            {
                "id": "run_demo_001",
                "name": "CIFAR-10 MobileNet baseline tuning",
                "dataset": "cifar10",
                "model_name": "mobilenet_v2",
                "status": "active",
                "experiment_count": 3,
            }
        ],
    )


def has_running_experiment(runs: list[dict[str, Any]]) -> bool:
    """Return whether any run has a running experiment."""
    for run in runs:
        run_detail = request_json(f'/runs/{run["id"]}', {})
        for experiment in run_detail.get("experiments", []):
            if experiment.get("status") == "running":
                return True
    return False


def load_run_detail(run_id: str) -> dict[str, Any]:
    """Return one run detail."""
    return request_json(
        f"/runs/{run_id}",
        {
            "id": run_id,
            "name": "CIFAR-10 MobileNet baseline tuning",
            "dataset": "cifar10",
            "model_name": "mobilenet_v2",
            "status": "active",
            "notes": "Using Streamlit fallback demo data because backend is not reachable.",
            "baseline_experiment_id": "exp_demo_001",
            "best_experiment_id": "exp_demo_002",
            "frontier_experiment_id": "exp_demo_002",
            "experiments": [
                {"id": "exp_demo_001", "run_id": run_id, "status": "success", "model_name": "mobilenet_v2", "decision": "keep", "is_best_so_far": False},
                {"id": "exp_demo_002", "run_id": run_id, "status": "success", "model_name": "mobilenet_v2", "decision": "keep", "is_best_so_far": True},
                {"id": "exp_demo_003", "run_id": run_id, "status": "success", "model_name": "mobilenet_v2", "decision": "discard", "is_best_so_far": False},
            ],
        },
    )


def load_run_summary(run_id: str) -> dict[str, Any]:
    """Return one run-level summary."""
    return request_json(
        f"/runs/{run_id}/summary",
        {
            "run_id": run_id,
            "baseline_experiment_id": "exp_demo_001",
            "best_experiment_id": "exp_demo_002",
            "frontier_experiment_id": "exp_demo_002",
            "keep_count": 2,
            "discard_count": 1,
            "crash_count": 0,
            "timeout_count": 0,
        },
    )


def load_run_metrics(run_id: str, metric_name: str) -> dict[str, Any]:
    """Return trend metrics for a run."""
    return request_json(
        f"/runs/{run_id}/metrics?metric_name={metric_name}",
        {
            "run_id": run_id,
            "metric_name": metric_name,
            "available_metrics": ["top1_acc", "val_loss", "train_loss"],
            "points": [
                {"experiment_id": "exp_demo_001", "experiment_index": 1, "metric_name": metric_name, "metric_value": 0.71},
                {"experiment_id": "exp_demo_002", "experiment_index": 2, "metric_name": metric_name, "metric_value": 0.78},
                {"experiment_id": "exp_demo_003", "experiment_index": 3, "metric_name": metric_name, "metric_value": 0.80},
            ],
        },
    )


def load_experiment_detail(experiment_id: str) -> dict[str, Any]:
    """Return one experiment detail."""
    return request_json(
        f"/experiments/{experiment_id}",
        {
            "id": experiment_id,
            "run_id": "run_demo_001",
            "status": "success",
            "decision": "keep",
            "decision_reason": "当前 run 下的最佳实验。",
            "baseline_experiment_id": "exp_demo_001",
            "is_best_so_far": experiment_id == "exp_demo_002",
            "config": {
                "task_type": "classification",
                "dataset": "cifar10",
                "model_family": "mobilenet",
                "model_name": "mobilenet_v2",
                "parameter_space_version": "mobilenet_v2@v1",
                "participates_in_ranking": True,
                "params": {
                    "optimizer": "adamw",
                    "learning_rate": 0.004,
                    "batch_size": 128,
                    "image_size": 64,
                    "epochs": 30,
                    "weight_decay": 0.0001,
                    "scheduler": "cosine",
                    "augmentation_policy": "basic",
                    "label_smoothing": 0.08,
                    "aux_logits": None,
                },
            },
            "parameter_space": {
                "model_name": "mobilenet_v2",
                "version": "mobilenet_v2@v1",
                "editable_params": {},
            },
            "proposal": {
                "task_type": "classification",
                "model_name": "mobilenet_v2",
                "based_on_experiment_ids": ["exp_demo_001"],
                "hypothesis": "Slightly higher learning rate may improve early convergence.",
                "changes": {"learning_rate": 0.004, "label_smoothing": 0.08},
                "reason": "The previous baseline is still underfitting in early epochs.",
                "risk": "low",
            },
            "result": {
                "status": "success",
                "metrics": {"train_loss": 0.42, "val_loss": 0.51, "top1_acc": 0.78, "best_epoch": 24},
                "resource": {"gpu_memory_mb": 2100, "training_seconds": 320},
                "params": {
                    "optimizer": "adamw",
                    "learning_rate": 0.004,
                    "batch_size": 128,
                    "image_size": 64,
                    "epochs": 30,
                    "weight_decay": 0.0001,
                    "scheduler": "cosine",
                    "augmentation_policy": "basic",
                    "label_smoothing": 0.08,
                    "aux_logits": None,
                },
                "artifacts": {
                    "log_path": "artifacts/runs/run_demo_001.log",
                    "checkpoint_path": "artifacts/checkpoints/exp_demo_002.pt",
                },
            },
            "reflection": {
                "outcome": "improved",
                "analysis": "Higher learning rate improved convergence without obvious instability.",
                "confidence": 0.78,
                "next_action": "Explore nearby learning rates and keep cosine scheduler.",
                "recommended_changes": {"learning_rate": 0.005},
            },
        },
    )


def append_activity_log(message: str) -> None:
    """Append one timestamped activity message to session state."""
    activity_logs = st.session_state.setdefault("activity_logs", [])
    activity_logs.append(f'[{time.strftime("%H:%M:%S")}] {message}')
    st.session_state["activity_logs"] = activity_logs[-LOG_LIMIT:]
    refresh_activity_log_view()


def refresh_activity_log_view() -> None:
    """Refresh the live activity log area when available."""
    global LIVE_LOG_CONTAINER
    if LIVE_LOG_CONTAINER is None:
        return
    log_lines = st.session_state.get("activity_logs", [])
    log_text = "\n".join(log_lines[-20:]) if log_lines else "Logs will appear here once training starts."
    LIVE_LOG_CONTAINER.code(log_text, language=None, wrap_lines=True, height=220)


def format_metric_summary(experiment_detail: dict[str, Any]) -> str:
    """Build a compact metric summary for logs."""
    metrics = (experiment_detail.get("result") or {}).get("metrics") or {}
    top1_acc = metrics.get("top1_acc")
    val_loss = metrics.get("val_loss")
    train_loss = metrics.get("train_loss")
    best_epoch = metrics.get("best_epoch")
    metric_parts = []
    if top1_acc is not None:
        metric_parts.append(f"top1_acc={top1_acc}")
    if val_loss is not None:
        metric_parts.append(f"val_loss={val_loss}")
    if train_loss is not None:
        metric_parts.append(f"train_loss={train_loss}")
    if best_epoch is not None:
        metric_parts.append(f"best_epoch={best_epoch}")
    return ", ".join(metric_parts) if metric_parts else "no metrics returned"


def format_proposal_changes(changes: dict[str, Any]) -> str:
    """Format one proposal change payload for display."""
    visible_changes = [
        f"{key}={value}"
        for key, value in changes.items()
        if key not in AI_BLOCKED_CHANGE_FIELDS and value is not None
    ]
    return ", ".join(visible_changes) if visible_changes else "无参数变更"


def clear_database_records() -> tuple[bool, Any]:
    """Clear all backend records."""
    return post_without_body("/runs/reset")


def clear_selected_run_records(run_id: str) -> tuple[bool, Any]:
    """Clear the currently selected run and all of its records."""
    return post_without_body(f"/runs/{run_id}/reset")


def stop_current_experiment(experiment_id: str) -> tuple[bool, Any]:
    """Stop the current running experiment immediately."""
    return post_without_body(f"/experiments/{experiment_id}/stop")


def reset_frontend_state_after_clear() -> None:
    """Clear session state that mirrors backend records."""
    preserved_keys = {
        "run_name",
        "dataset",
        "model_name",
        "optimizer",
        "learning_rate",
        "batch_size",
        "image_size",
        "epochs",
        "weight_decay",
        "scheduler",
        "augmentation_policy",
        "label_smoothing",
        "aux_logits",
        "auto_train_time_budget_minutes",
    }
    preserved_values = {key: st.session_state.get(key) for key in preserved_keys if key in st.session_state}
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state.update(preserved_values)
    st.session_state["selected_run_id"] = "__all__"
    st.session_state["activity_logs"] = []


def store_ai_panel(payload: dict[str, Any]) -> None:
    """Store the latest AI panel payload."""
    st.session_state["manual_ai_panel"] = payload
    st.session_state["active_ai_panel_mode"] = "manual"
    refresh_ai_panel_view()


def build_result_snapshot(experiment_id: str, experiment_detail: dict[str, Any]) -> dict[str, Any]:
    """Build one compact result snapshot."""
    metrics = (experiment_detail.get("result") or {}).get("metrics") or {}
    return {
        "experiment_id": experiment_id,
        "status": experiment_detail.get("status"),
        "metrics": metrics,
        "summary": format_metric_summary(experiment_detail),
    }


def store_manual_ai_suggestion(
    run_id: str,
    experiment_id: str,
    experiment_detail: dict[str, Any],
    proposal: dict[str, Any],
) -> None:
    """Store one post-train suggestion for manual train mode."""
    store_ai_panel(
        {
            "mode": "manual",
            "run_id": run_id,
            "experiment_id": experiment_id,
            "result": build_result_snapshot(experiment_id, experiment_detail),
            "proposal": proposal,
        }
    )


def refresh_ai_panel_view() -> None:
    """Refresh the live AI panel when available."""
    global LIVE_AI_PANEL_CONTAINER
    if LIVE_AI_PANEL_CONTAINER is None:
        return
    active_mode = st.session_state.get("active_ai_panel_mode")
    if active_mode == "auto":
        suggestion_payload = st.session_state.get("auto_train_summary")
    elif active_mode == "manual":
        suggestion_payload = st.session_state.get("manual_ai_panel")
    else:
        suggestion_payload = st.session_state.get("auto_train_summary") or st.session_state.get("manual_ai_panel")
    with LIVE_AI_PANEL_CONTAINER.container():
        with st.container(border=True):
            if not suggestion_payload:
                st.caption("AI suggestions and tuning trends will appear here during or after training.")
                st.code("Waiting for training output...", language=None, wrap_lines=True, height=180)
                return
            if suggestion_payload.get("mode") == "auto":
                final_proposal = suggestion_payload.get("final_proposal")
                baseline = suggestion_payload.get("baseline", {})
                rounds = suggestion_payload.get("rounds", [])
                task_status = suggestion_payload.get("task_status")
                stop_reason = suggestion_payload.get("stop_reason")
                progress = st.session_state.get("auto_task_progress") or {}
                st.markdown("**Auto Train Summary**")
                if progress and progress.get("status") in {"queued", "running", "stopping"}:
                    completed_rounds = len(rounds)
                    current_round = progress.get("current_round", 0)
                    elapsed_seconds = progress.get("elapsed_seconds", 0.0)
                    max_wall_clock_minutes = progress.get("max_wall_clock_minutes", 0)
                    current_experiment_id = progress.get("current_experiment_id") or "-"
                    st.caption(
                        f"进度：已完成 {completed_rounds} 轮，"
                        f"当前轮次 {current_round}，"
                        f"已用 {format_elapsed_seconds(elapsed_seconds)} / {max_wall_clock_minutes}m，"
                        f"实验 {get_short_experiment_id(current_experiment_id)}。"
                    )
                st.markdown(f"基线实验：`{get_short_experiment_id(baseline.get('experiment_id'))}`  ")
                st.caption(baseline.get("summary", ""))
                trend_rows = []
                baseline_metrics = baseline.get("metrics", {})
                trend_rows.append(
                    {
                        "round_index": 0,
                        "top1_acc": baseline_metrics.get("top1_acc"),
                        "val_loss": baseline_metrics.get("val_loss"),
                        "train_loss": baseline_metrics.get("train_loss"),
                    }
                )
                for round_info in rounds:
                    round_metrics = round_info["result"].get("metrics", {})
                    st.markdown(
                        f"第 {round_info['round_index']} 轮："
                        f"{format_proposal_changes(round_info['proposal']['changes'])}"
                    )
                    st.caption(
                        f"{get_short_experiment_id(round_info['result']['experiment_id'])} | "
                        f"{round_info['result']['summary']}"
                    )
                    trend_rows.append(
                        {
                            "round_index": int(round_info["round_index"]),
                            "top1_acc": round_metrics.get("top1_acc"),
                            "val_loss": round_metrics.get("val_loss"),
                            "train_loss": round_metrics.get("train_loss"),
                        }
                    )
                if len(trend_rows) > 0:
                    st.markdown("**Tuning Trend**")
                    st.line_chart(
                        trend_rows,
                        x="round_index",
                        y=["top1_acc", "val_loss", "train_loss"],
                        use_container_width=True,
                    )
                if final_proposal:
                    st.markdown("**Next Suggestion**")
                    st.markdown(final_proposal["hypothesis"])
                    st.caption(final_proposal["reason"])
                    st.markdown(f"`{format_proposal_changes(final_proposal['changes'])}`")
                elif task_status in {"stopped", "stopped_by_budget", "stopped_by_policy", "failed"}:
                    st.caption(stop_reason or "自动训练已经结束。")
                else:
                    st.caption("自动训练进行中。每完成一轮后，这里的趋势会自动更新。")
                return

            proposal = suggestion_payload["proposal"]
            result = suggestion_payload["result"]
            st.markdown("**Single-Run Suggestion**")
            st.caption(f"实验 {get_short_experiment_id(result['experiment_id'])} | {result['summary']}")
            st.markdown(f"**Suggestion**  \n{proposal['hypothesis']}")
            st.caption(proposal["reason"])
            st.markdown(f"**Suggested Changes**  \n{format_proposal_changes(proposal['changes'])}")


def generate_and_store_ai_suggestion(
    run_id: str,
    experiment_id: str,
    experiment_detail: dict[str, Any],
    log_prefix: str,
) -> tuple[bool, dict[str, Any] | str]:
    """Generate one AI suggestion for a completed experiment and store it."""
    ok, proposal_response = generate_aihubmix_proposal_request(run_id)
    if not ok:
        error_message = f'AI suggestion failed: {proposal_response.get("detail", proposal_response)}'
        append_activity_log(error_message)
        return False, error_message
    append_activity_log(f"{log_prefix} AI suggestion: {proposal_response['hypothesis']}")
    append_activity_log(f"{log_prefix} Proposed changes: {format_proposal_changes(proposal_response['changes'])}")
    store_manual_ai_suggestion(run_id, experiment_id, experiment_detail, proposal_response)
    return True, proposal_response


def set_ui_locked(is_locked: bool) -> None:
    """Lock or unlock all train actions."""
    st.session_state["ui_locked"] = is_locked


def set_post_action_notice(message: str, level: str = "success") -> None:
    """Store one notice to be shown after the next rerun."""
    st.session_state["post_action_notice"] = {"message": message, "level": level}


def clear_training_state() -> None:
    """Clear all transient frontend training state after completion."""
    st.session_state["ui_locked"] = False
    st.session_state["active_train_control"] = None
    st.session_state["training_experiment_id"] = None
    st.session_state["last_running_experiment_id"] = None
    st.session_state["current_auto_task_id"] = None
    st.session_state["auto_task_progress"] = None
    st.session_state["skip_auto_poll_once"] = False


def sync_auto_train_task_state() -> str | None:
    """Sync frontend state from one backend auto-train task snapshot."""
    current_auto_task_id = st.session_state.get("current_auto_task_id")
    if not current_auto_task_id:
        return None

    st.session_state["active_train_control"] = "auto"
    ok, auto_task_response = load_auto_train_task(current_auto_task_id)
    if not ok or "task_id" not in auto_task_response:
        return None

    st.session_state["activity_logs"] = auto_task_response.get("logs", [])
    summary = auto_task_response.get("summary")
    if summary is not None:
        summary = dict(summary)
        summary["task_status"] = auto_task_response.get("status")
        summary["stop_reason"] = auto_task_response.get("stop_reason")
        st.session_state["auto_train_summary"] = summary
        st.session_state["active_ai_panel_mode"] = "auto"
        completed_rounds = len(summary.get("rounds", [])) if summary.get("mode") == "auto" else 0
        if completed_rounds != st.session_state.get("last_auto_completed_rounds", -1):
            st.session_state["last_auto_completed_rounds"] = completed_rounds
            st.session_state["auto_result_refresh_needed"] = True
    run_id = auto_task_response.get("run_id")
    if run_id:
        st.session_state["selected_run_id"] = run_id
    st.session_state["training_experiment_id"] = auto_task_response.get("current_experiment_id")
    st.session_state["auto_task_progress"] = {
        "status": auto_task_response.get("status"),
        "current_round": auto_task_response.get("current_round", 0),
        "max_wall_clock_minutes": auto_task_response.get("max_wall_clock_minutes", 0),
        "elapsed_seconds": auto_task_response.get("elapsed_seconds", 0.0),
        "current_experiment_id": auto_task_response.get("current_experiment_id"),
    }
    if run_id:
        run_summary = load_run_summary(run_id)
        best_experiment_id = run_summary.get("best_experiment_id")
        if best_experiment_id and best_experiment_id != st.session_state.get("selected_experiment_id"):
            st.session_state["selected_experiment_id"] = best_experiment_id
            st.session_state["auto_result_refresh_needed"] = True

    auto_status = auto_task_response.get("status")
    if auto_status in {"queued", "running", "stopping"}:
        st.session_state["ui_locked"] = True
    return auto_status


@st.fragment(run_every="2s")
def render_live_training_monitor() -> None:
    """Keep the log and AI panel updated while training is active."""
    auto_task_id = st.session_state.get("current_auto_task_id")
    experiment_id = st.session_state.get("training_experiment_id")
    if not auto_task_id and not experiment_id:
        return

    if auto_task_id:
        auto_status = sync_auto_train_task_state()
        refresh_activity_log_view()
        refresh_ai_panel_view()
        if st.session_state.pop("auto_result_refresh_needed", False):
            st.rerun()
        if auto_status in {"stopped", "stopped_by_budget", "stopped_by_policy", "failed"}:
            task_snapshot = request_json(
                f"/runs/auto-train/{auto_task_id}",
                {"error": "unknown error"},
            )
            if auto_status == "stopped":
                set_post_action_notice("Auto Train stopped and discarded the current experiment.", "success")
            elif auto_status in {"stopped_by_budget", "stopped_by_policy"}:
                set_post_action_notice(
                    task_snapshot.get("stop_reason", "Auto Train stopped."),
                    "success",
                )
            elif auto_status == "failed":
                set_post_action_notice(
                    f"Auto Train failed: {task_snapshot.get('error', 'unknown error')}",
                    "error",
                )
            clear_training_state()
            st.rerun()
        return

    experiment_detail = load_experiment_detail(experiment_id)
    if experiment_detail.get("status") in {"success", "failed", "discarded"}:
        st.rerun()


def queue_train_request(payload: dict[str, Any]) -> None:
    """Queue one manual train request and force a rerun with locked UI."""
    if st.session_state.get("ui_locked") or st.session_state.get("pending_train_request"):
        return
    st.session_state["active_train_control"] = "manual"
    st.session_state["pending_train_request"] = {
        "payload": payload,
        "selected_run_id": st.session_state.get("selected_run_id", "__all__"),
    }
    set_ui_locked(True)
    st.rerun()


def process_pending_train_request() -> None:
    """Execute one queued manual train request."""
    pending_request = st.session_state.get("pending_train_request")
    if not pending_request:
        return

    st.session_state["pending_train_request"] = None
    payload = pending_request["payload"]
    selected_run_id = pending_request.get("selected_run_id", "__all__")

    append_activity_log("Manual train requested from the current parameter panel.")
    if selected_run_id == "__all__":
        ok, message = create_run_and_first_experiment(payload)
    else:
        latest_experiment = get_latest_experiment_detail(selected_run_id)
        payload["based_on_experiment_ids"] = [latest_experiment["id"]] if latest_experiment else []
        payload["proposal_hypothesis"] = "Manual follow-up experiment under the selected run."
        payload["proposal_reason"] = "Use the current parameter panel as the next structured experiment."
        ok, message = append_experiment_to_run(selected_run_id, payload)
    if ok:
        set_post_action_notice(message, "success")
        st.rerun()

    set_ui_locked(False)
    append_activity_log(message)
    st.error(message)


def build_all_training_records(runs: list[dict[str, Any]], selected_run_id: str) -> list[dict[str, Any]]:
    """Build training records for one run or all runs."""
    records: list[dict[str, Any]] = []
    candidate_runs = runs if selected_run_id == "__all__" else [run for run in runs if run["id"] == selected_run_id]
    for run in candidate_runs:
        run_detail = load_run_detail(run["id"])
        run_summary = load_run_summary(run["id"])
        for experiment in run_detail.get("experiments", []):
            detail = load_experiment_detail(experiment["id"])
            result = detail.get("result") or {}
            metrics = result.get("metrics") or {}
            params = detail.get("config", {}).get("params", {})
            anchor_labels = []
            if experiment["id"] == run_summary.get("baseline_experiment_id"):
                anchor_labels.append("baseline")
            if experiment["id"] == run_summary.get("best_experiment_id"):
                anchor_labels.append("best")
            if experiment["id"] == run_summary.get("frontier_experiment_id"):
                anchor_labels.append("frontier")
            records.append(
                {
                    "selected": True,
                    "run_id": run["id"],
                    "run_name": run["name"],
                    "exp": get_short_experiment_id(experiment["id"]),
                    "experiment_id": experiment["id"],
                    "anchor": "/".join(anchor_labels),
                    "decision": detail.get("decision"),
                    "status": experiment["status"],
                    "model_name": run["model_name"],
                    "dataset": run["dataset"],
                    "top1_acc": metrics.get("top1_acc"),
                    "val_loss": metrics.get("val_loss"),
                    "train_loss": metrics.get("train_loss"),
                    "best_epoch": metrics.get("best_epoch"),
                    "optimizer": params.get("optimizer"),
                    "learning_rate": params.get("learning_rate"),
                    "batch_size": params.get("batch_size"),
                    "image_size": params.get("image_size"),
                    "epochs": params.get("epochs"),
                    "scheduler": params.get("scheduler"),
                }
            )
    return records


def get_latest_experiment_detail(run_id: str) -> dict[str, Any] | None:
    """Return the latest experiment detail for one run."""
    run_detail = load_run_detail(run_id)
    experiments = run_detail.get("experiments", [])
    if not experiments:
        return None
    latest_experiment = experiments[-1]
    return load_experiment_detail(latest_experiment["id"])


def get_best_experiment_detail(run_id: str) -> dict[str, Any] | None:
    """Return the best finished experiment detail for one run."""
    run_summary = load_run_summary(run_id)
    best_experiment_id = run_summary.get("best_experiment_id")
    if best_experiment_id:
        return load_experiment_detail(best_experiment_id)

    run_detail = load_run_detail(run_id)
    experiments = run_detail.get("experiments", [])
    if not experiments:
        return None

    best_detail: dict[str, Any] | None = None
    best_key: tuple[float, float, int] | None = None
    for index, experiment in enumerate(experiments):
        detail = load_experiment_detail(experiment["id"])
        if detail.get("status") != "success":
            continue
        metrics = (detail.get("result") or {}).get("metrics") or {}
        top1_acc = metrics.get("top1_acc")
        val_loss = metrics.get("val_loss")
        ranking_key = (
            float(top1_acc) if top1_acc is not None else float("-inf"),
            -float(val_loss) if val_loss is not None else float("-inf"),
            float(index),
        )
        if best_key is None or ranking_key > best_key:
            best_key = ranking_key
            best_detail = detail

    return best_detail or get_latest_experiment_detail(run_id)


def load_reference_config(selected_run_id: str) -> dict[str, Any]:
    """Load the latest config for the selected run or return defaults."""
    default_config = {
        "run_name": build_auto_run_name("neu-cls", "mobilenet_v2"),
        "dataset": "neu-cls",
        "model_name": "mobilenet_v2",
        "participates_in_ranking": True,
        "search_policy": default_search_policy(),
        "ranking_policy": default_ranking_policy(),
        "params": {
            "optimizer": "adamw",
            "learning_rate": 0.003,
            "batch_size": 128,
            "image_size": get_original_image_size("neu-cls", "mobilenet_v2"),
            "epochs": 10,
            "weight_decay": 0.0001,
            "scheduler": "cosine",
            "augmentation_policy": "basic",
            "augmentation_params": {
                "mixup_alpha": 0.0,
                "cutmix_alpha": 0.0,
                "random_erasing_prob": 0.0,
            },
            "loss_name": "cross_entropy_with_label_smoothing",
            "loss_params": {"focal_gamma": 2.0},
            "label_smoothing": 0.1,
            "aux_logits": False,
        },
    }
    if selected_run_id == "__all__":
        return default_config
    latest_experiment = get_latest_experiment_detail(selected_run_id)
    if latest_experiment is None:
        return default_config
    run_detail = load_run_detail(selected_run_id)
    return {
        "run_name": build_auto_run_name(
            latest_experiment["config"]["dataset"],
            latest_experiment["config"]["model_name"],
        ),
        "dataset": latest_experiment["config"]["dataset"],
        "model_name": latest_experiment["config"]["model_name"],
        "participates_in_ranking": latest_experiment["config"].get("participates_in_ranking", True),
        "search_policy": latest_experiment["config"].get("search_policy") or default_search_policy(),
        "ranking_policy": latest_experiment["config"].get("ranking_policy") or default_ranking_policy(),
        "params": latest_experiment["config"]["params"],
    }


def build_payload_from_form(form_values: dict[str, Any]) -> dict[str, Any]:
    """Build backend payload from current form values."""
    model_name = form_values["model_name"]
    model_config = SUPPORTED_MODELS[model_name]
    parameter_space = deepcopy(model_config["parameter_space"])
    parameter_space["editable_params"]["image_size"]["choices"] = get_dataset_image_size_options(form_values["dataset"])
    parameter_space["editable_params"]["image_size"]["choices"] = get_image_size_search_choices(
        form_values["dataset"],
        model_name,
        int(form_values["image_size"]),
    )
    params = {
        "optimizer": form_values["optimizer"],
        "learning_rate": float(form_values["learning_rate"]),
        "batch_size": int(form_values["batch_size"]),
        "image_size": int(form_values["image_size"]),
        "epochs": int(form_values["epochs"]),
        "weight_decay": float(form_values["weight_decay"]),
        "scheduler": form_values["scheduler"],
        "augmentation_policy": form_values["augmentation_policy"],
        "augmentation_params": {
            "mixup_alpha": 0.0,
            "cutmix_alpha": 0.0,
            "random_erasing_prob": 0.0,
        },
        "loss_name": "cross_entropy_with_label_smoothing",
        "loss_params": {
            "focal_gamma": 2.0,
        },
        "label_smoothing": float(form_values["label_smoothing"]),
        "aux_logits": bool(form_values["aux_logits"]) if model_name == "googlenet" else None,
    }
    return {
        "name": form_values["run_name"],
        "dataset": form_values["dataset"],
        "model_name": model_name,
        "notes": form_values.get("notes", ""),
        "proposal_hypothesis": form_values.get("proposal_hypothesis", ""),
        "proposal_reason": form_values.get("proposal_reason", ""),
        "based_on_experiment_ids": form_values.get("based_on_experiment_ids", []),
        "proposal_changes": params,
        "config": {
            "task_type": "classification",
            "dataset": form_values["dataset"],
            "model_family": model_config["model_family"],
            "model_name": model_name,
            "parameter_space_version": model_config["parameter_space_version"],
            "participates_in_ranking": bool(form_values.get("participates_in_ranking", True)),
            "search_policy": form_values["search_policy"],
            "ranking_policy": form_values["ranking_policy"],
            "params": params,
        },
        "parameter_space": parameter_space,
    }


def create_run_and_first_experiment(payload: dict[str, Any]) -> tuple[bool, str]:
    """Create a run, then create and train its first experiment."""
    run_request = {
        "name": payload["name"],
        "dataset": payload["dataset"],
        "model_name": payload["model_name"],
        "base_config": payload["config"],
        "notes": payload["notes"],
    }
    ok, run_response = post_json("/runs", run_request)
    if not ok:
        return False, f'Create run failed: {run_response.get("detail", run_response)}'
    append_activity_log(f"Created run {run_response['id']} ({payload['model_name']} on {payload['dataset']}).")

    experiment_request = {
        "run_id": run_response["id"],
        "config": payload["config"],
        "parameter_space": payload["parameter_space"],
        "proposal": None,
    }
    ok, experiment_response = post_json("/experiments", experiment_request)
    if not ok:
        return False, f'Create experiment failed: {experiment_response.get("detail", experiment_response)}'
    append_activity_log(f"Created experiment {experiment_response['id']} as the baseline trial.")

    ok, train_response = post_without_body(f'/experiments/{experiment_response["id"]}/train')
    if not ok:
        return False, f'Train experiment failed: {train_response.get("detail", train_response)}'
    append_activity_log(f"Started training experiment {experiment_response['id']}.")

    st.session_state["selected_run_id"] = run_response["id"]
    st.session_state["selected_experiment_id"] = experiment_response["id"]
    st.session_state["training_experiment_id"] = experiment_response["id"]
    return True, f'Started training for run {run_response["id"]} and experiment {experiment_response["id"]}.'


def append_experiment_to_run(run_id: str, payload: dict[str, Any]) -> tuple[bool, str]:
    """Append one experiment to an existing run and start training."""
    experiment_request = {
        "run_id": run_id,
        "config": payload["config"],
        "parameter_space": payload["parameter_space"],
        "proposal": {
            "task_type": "classification",
            "model_name": payload["model_name"],
            "based_on_experiment_ids": payload["based_on_experiment_ids"],
            "hypothesis": payload["proposal_hypothesis"],
            "changes": payload["proposal_changes"],
            "reason": payload["proposal_reason"],
            "risk": "low",
        },
    }
    ok, experiment_response = post_json("/experiments", experiment_request)
    if not ok:
        return False, f'Append experiment failed: {experiment_response.get("detail", experiment_response)}'
    append_activity_log(
        f"Appended experiment {experiment_response['id']} to run {run_id} with changes: "
        f"{format_proposal_changes(payload['proposal_changes'])}."
    )

    ok, train_response = post_without_body(f'/experiments/{experiment_response["id"]}/train')
    if not ok:
        return False, f'Train experiment failed: {train_response.get("detail", train_response)}'
    append_activity_log(f"Started training experiment {experiment_response['id']}.")

    st.session_state["selected_run_id"] = run_id
    st.session_state["selected_experiment_id"] = experiment_response["id"]
    st.session_state["training_experiment_id"] = experiment_response["id"]
    return True, f'Appended and started experiment {experiment_response["id"]} under run {run_id}.'


def render_control_panel(runs: list[dict[str, Any]], selected_run_id: str, is_training_active: bool) -> None:
    """Render the left-side parameter and action panel."""
    st.subheader("Train")
    st.caption("这里用于发起单次训练或自动连续调优。选中已有 run 时，新实验会追加到当前 run。")
    if is_training_active:
        st.warning("A training job is running. Actions are temporarily locked.")

    reference_config = load_reference_config(selected_run_id)
    if st.session_state.get("form_reference_run") != selected_run_id:
        st.session_state["run_name"] = reference_config["run_name"]
        st.session_state["dataset"] = reference_config["dataset"]
        st.session_state["model_name"] = reference_config["model_name"]
        st.session_state["optimizer"] = reference_config["params"]["optimizer"]
        st.session_state["learning_rate"] = reference_config["params"]["learning_rate"]
        st.session_state["batch_size"] = reference_config["params"]["batch_size"]
        st.session_state["image_size"] = reference_config["params"]["image_size"]
        st.session_state["epochs"] = reference_config["params"]["epochs"]
        st.session_state["weight_decay"] = reference_config["params"]["weight_decay"]
        st.session_state["scheduler"] = reference_config["params"]["scheduler"]
        st.session_state["augmentation_policy"] = reference_config["params"]["augmentation_policy"]
        st.session_state["label_smoothing"] = reference_config["params"]["label_smoothing"]
        st.session_state["aux_logits"] = bool(reference_config["params"].get("aux_logits") or False)
        search_policy = reference_config.get("search_policy") or default_search_policy()
        st.session_state["allow_basic_hparam_search"] = search_policy["allow_basic_hparam_search"]
        st.session_state["allowed_basic_hparam_fields"] = search_policy.get(
            "allowed_basic_hparam_fields",
            get_allowed_basic_hparam_fields(
                reference_config["dataset"],
                reference_config["model_name"],
                int(reference_config["params"]["image_size"]),
            ),
        )
        st.session_state["allow_strategy_search"] = search_policy["allow_strategy_search"]
        st.session_state["allow_loss_search"] = search_policy["allow_loss_search"]
        st.session_state["allow_augmentation_search"] = search_policy["allow_augmentation_search"]
        st.session_state["require_manual_approval_for_high_impact_changes"] = search_policy[
            "require_manual_approval_for_high_impact_changes"
        ]
        ranking_policy = reference_config.get("ranking_policy") or default_ranking_policy()
        default_policy = default_ranking_policy()
        st.session_state["use_default_ranking_policy"] = ranking_policy == default_policy
        st.session_state["ranking_primary_metric"] = ranking_policy.get("primary_metric", default_policy["primary_metric"])
        st.session_state["ranking_min_primary_metric_improvement"] = float(
            ranking_policy.get("min_primary_metric_improvement", default_policy["min_primary_metric_improvement"])
        )
        st.session_state["ranking_primary_metric_parity_epsilon"] = float(
            ranking_policy.get("primary_metric_parity_epsilon", default_policy["primary_metric_parity_epsilon"])
        )
        st.session_state["ranking_tie_breaker_metric"] = ranking_policy.get(
            "tie_breaker_metric",
            default_policy["tie_breaker_metric"],
        )
        st.session_state["ranking_min_tie_breaker_metric_improvement"] = float(
            ranking_policy.get(
                "min_tie_breaker_metric_improvement",
                default_policy["min_tie_breaker_metric_improvement"],
            )
        )
        st.session_state["ranking_max_image_size"] = ranking_policy.get("max_image_size")
        st.session_state["based_on_experiment_ids"] = []
        st.session_state["form_reference_run"] = selected_run_id

    original_image_size = get_original_image_size(st.session_state["dataset"], st.session_state["model_name"])
    if selected_run_id == "__all__":
        st.session_state["image_size"] = original_image_size
    allowed_ranking_image_sizes = [None] + get_dataset_image_size_options(st.session_state["dataset"])
    if st.session_state.get("ranking_max_image_size") not in allowed_ranking_image_sizes:
        st.session_state["ranking_max_image_size"] = None
    if "auto_train_time_budget_minutes" not in st.session_state:
        st.session_state["auto_train_time_budget_minutes"] = 60

    with st.container(border=True):
        st.markdown("**Training Setup**")
        st.caption("这里配置本轮训练的基线参数。切换到已有 run 时，会自动带入该 run 最近一次实验的配置。")
        top_left, top_mid, top_right = st.columns(3)
        with top_left:
            st.text_input("Run Name", key="run_name")
        with top_mid:
            st.selectbox("Dataset", options=SUPPORTED_DATASETS, key="dataset", on_change=sync_auto_run_name)
        with top_right:
            model_name = st.selectbox(
                "Model",
                options=["mobilenet_v2", "googlenet", "resnet18", "resnet34", "densenet121"],
                format_func=get_model_label,
                key="model_name",
                on_change=sync_auto_run_name,
            )

        row_one = st.columns(3)
        with row_one[0]:
            st.number_input("Learning Rate", min_value=0.0001, max_value=0.01, step=0.0001, format="%.4f", key="learning_rate")
        with row_one[1]:
            st.selectbox("Batch Size", options=[32, 64, 128, 256], key="batch_size")
        with row_one[2]:
            st.selectbox("Epochs", options=[10, 20, 30, 50], key="epochs")

        row_two = st.columns(3)
        with row_two[0]:
            st.selectbox("Image Size", options=get_dataset_image_size_options(st.session_state["dataset"]), key="image_size")
        with row_two[1]:
            auto_train_time_budget_minutes = st.number_input(
                "Auto Train Budget (min)",
                min_value=1,
                max_value=24 * 60,
                step=5,
                key="auto_train_time_budget_minutes",
            )
        with row_two[2]:
            pass

        st.caption("Current Managed Params: " + summarize_ai_managed_params(st.session_state))

        with st.expander("AI Search Policy", expanded=False):
            image_size_choices = get_image_size_search_choices(
                st.session_state["dataset"],
                model_name,
                int(st.session_state["image_size"]),
            )
            allowed_basic_hparam_fields = get_allowed_basic_hparam_fields(
                st.session_state["dataset"],
                st.session_state["model_name"],
                int(st.session_state["image_size"]),
            )
            st.caption(
                summarize_effective_ai_search_fields(
                    st.session_state["allow_basic_hparam_search"],
                    allowed_basic_hparam_fields,
                    st.session_state["allow_strategy_search"],
                    st.session_state["allow_loss_search"],
                    st.session_state["allow_augmentation_search"],
                )
            )
            st.caption(
                f"补充说明：image_size 保持原图大小时不搜索；当你手动改成非原图大小时，只在原图大小和当前设置之间搜索。"
                f"当前原图大小：{original_image_size}；当前 image_size 搜索范围："
                + ", ".join(str(choice) for choice in image_size_choices)
            )
            st.checkbox(
                "Allow basic hyperparameter search",
                key="allow_basic_hparam_search",
                disabled=is_training_active,
            )
            st.checkbox("Allow strategy search", key="allow_strategy_search", disabled=is_training_active)
            st.checkbox("Allow loss search", key="allow_loss_search", disabled=is_training_active)
            st.checkbox("Allow augmentation search", key="allow_augmentation_search", disabled=is_training_active)
            st.checkbox(
                "Require manual approval for high-impact changes",
                key="require_manual_approval_for_high_impact_changes",
                disabled=is_training_active,
            )

        with st.expander("Ranking Policy", expanded=False):
            st.caption("这里控制 run 内实验如何晋级。默认模式保持轻量；关闭默认后可微调主指标、灰区和平局裁决。")
            st.checkbox(
                "Use default ranking policy",
                key="use_default_ranking_policy",
                disabled=is_training_active,
            )
            if st.session_state["use_default_ranking_policy"]:
                st.caption("Default: " + summarize_ranking_policy(default_ranking_policy()))
            else:
                ranking_columns = st.columns(2)
                with ranking_columns[0]:
                    st.selectbox(
                        "Primary Metric",
                        options=["top1_acc", "val_loss"],
                        format_func=lambda metric: RANKING_METRIC_LABELS.get(metric, metric),
                        key="ranking_primary_metric",
                        disabled=is_training_active,
                    )
                    st.number_input(
                        "Min Primary Improvement",
                        min_value=0.0,
                        max_value=1.0,
                        step=0.001,
                        format="%.4f",
                        key="ranking_min_primary_metric_improvement",
                        disabled=is_training_active,
                    )
                    st.number_input(
                        "Primary Parity Epsilon",
                        min_value=0.0,
                        max_value=1.0,
                        step=0.0001,
                        format="%.4f",
                        key="ranking_primary_metric_parity_epsilon",
                        disabled=is_training_active,
                    )
                with ranking_columns[1]:
                    st.selectbox(
                        "Tie Breaker",
                        options=["val_loss", "top1_acc", "training_seconds"],
                        format_func=lambda metric: RANKING_METRIC_LABELS.get(metric, metric),
                        key="ranking_tie_breaker_metric",
                        disabled=is_training_active,
                    )
                    st.number_input(
                        "Min Tie Breaker Improvement",
                        min_value=0.0,
                        max_value=10.0,
                        step=0.001,
                        format="%.4f",
                        key="ranking_min_tie_breaker_metric_improvement",
                        disabled=is_training_active,
                    )
                    st.selectbox(
                        "Max Image Size",
                        options=allowed_ranking_image_sizes,
                        format_func=lambda value: "No limit" if value is None else str(value),
                        key="ranking_max_image_size",
                        disabled=is_training_active,
                    )

    payload = build_payload_from_form(
        {
            "run_name": st.session_state["run_name"],
            "dataset": st.session_state["dataset"],
            "model_name": st.session_state["model_name"],
            "optimizer": st.session_state["optimizer"],
            "learning_rate": st.session_state["learning_rate"],
            "batch_size": st.session_state["batch_size"],
            "image_size": st.session_state["image_size"],
            "epochs": st.session_state["epochs"],
            "weight_decay": st.session_state["weight_decay"],
            "scheduler": st.session_state["scheduler"],
            "augmentation_policy": st.session_state["augmentation_policy"],
            "label_smoothing": st.session_state["label_smoothing"],
            "aux_logits": st.session_state["aux_logits"],
            "search_policy": {
                "allow_basic_hparam_search": st.session_state["allow_basic_hparam_search"],
                "allowed_basic_hparam_fields": get_allowed_basic_hparam_fields(
                    st.session_state["dataset"],
                    st.session_state["model_name"],
                    int(st.session_state["image_size"]),
                ),
                "allow_strategy_search": st.session_state["allow_strategy_search"],
                "allow_loss_search": st.session_state["allow_loss_search"],
                "allow_augmentation_search": st.session_state["allow_augmentation_search"],
                "require_manual_approval_for_high_impact_changes": st.session_state[
                    "require_manual_approval_for_high_impact_changes"
                ],
            },
            "ranking_policy": (
                default_ranking_policy()
                if st.session_state["use_default_ranking_policy"]
                else {
                    "primary_metric": st.session_state["ranking_primary_metric"],
                    "primary_metric_mode": get_ranking_metric_mode(st.session_state["ranking_primary_metric"]),
                    "min_primary_metric_improvement": float(st.session_state["ranking_min_primary_metric_improvement"]),
                    "primary_metric_parity_epsilon": float(st.session_state["ranking_primary_metric_parity_epsilon"]),
                    "tie_breaker_metric": st.session_state["ranking_tie_breaker_metric"],
                    "tie_breaker_mode": get_ranking_metric_mode(st.session_state["ranking_tie_breaker_metric"]),
                    "min_tie_breaker_metric_improvement": float(
                        st.session_state["ranking_min_tie_breaker_metric_improvement"]
                    ),
                    "max_image_size": st.session_state["ranking_max_image_size"],
                }
            ),
            "based_on_experiment_ids": st.session_state.get("based_on_experiment_ids", []),
        }
    )

    is_appending_to_existing_run = selected_run_id != "__all__"
    train_button_label = "Append Train" if is_appending_to_existing_run else "Train"
    auto_train_button_label = "Append Auto Train" if is_appending_to_existing_run else "Auto Train"
    active_train_control = st.session_state.get("active_train_control")
    action_left, action_right = st.columns(2)
    with action_left:
        left_label = "Stop Training" if is_training_active and active_train_control == "manual" else train_button_label
        left_disabled = is_training_active and active_train_control != "manual"
        if st.button(left_label, disabled=left_disabled, use_container_width=True):
            if is_training_active and active_train_control == "manual":
                running_experiment_id = st.session_state.get("training_experiment_id") or find_running_experiment_id(runs)
                if running_experiment_id:
                    ok, response = stop_current_experiment(running_experiment_id)
                    if ok:
                        clear_training_state()
                        append_activity_log(f"Experiment {running_experiment_id} was stopped and discarded.")
                        set_post_action_notice(f"Stopped and discarded experiment {running_experiment_id}.", "success")
                        st.rerun()
                    else:
                        st.error(f'Stop training failed: {response.get("detail", response)}')
            else:
                queue_train_request(payload)
    with action_right:
        right_label = "Stop Training" if is_training_active and active_train_control == "auto" else auto_train_button_label
        right_disabled = is_training_active and active_train_control != "auto"
        if st.button(right_label, disabled=right_disabled, use_container_width=True):
            if is_training_active and active_train_control == "auto":
                current_auto_task_id = st.session_state.get("current_auto_task_id")
                if current_auto_task_id:
                    ok, response = stop_auto_train_task_request(current_auto_task_id)
                    if ok:
                        append_activity_log(f"Stopping auto train task {current_auto_task_id}...")
                        st.rerun()
                    else:
                        st.error(f'Stop training failed: {response.get("detail", response)}')
            else:
                st.session_state["active_train_control"] = "auto"
                ok, response = start_auto_train_task_request(
                    payload,
                    selected_run_id,
                    int(auto_train_time_budget_minutes),
                )
                if ok:
                    st.session_state["current_auto_task_id"] = response["task_id"]
                    st.session_state["ui_locked"] = True
                    st.session_state["skip_auto_poll_once"] = True
                    append_activity_log(
                        f"Auto Train task started: {response['task_id']} | "
                        f"time budget={int(auto_train_time_budget_minutes)}m"
                    )
                    st.rerun()
                else:
                    st.session_state["active_train_control"] = None
                    st.error(f'Auto Train failed to start: {response.get("detail", response)}')


def render_activity_log() -> None:
    """Render the training activity log."""
    global LIVE_LOG_CONTAINER
    st.subheader("Execution Log")
    LIVE_LOG_CONTAINER = st.empty()
    refresh_activity_log_view()


def render_ai_suggestion_panel() -> None:
    """Render the latest AI suggestion card."""
    global LIVE_AI_PANEL_CONTAINER
    st.subheader("AI Suggestions")
    st.caption("这里展示单次训练建议、自动调优总结和下一步推荐动作。")
    LIVE_AI_PANEL_CONTAINER = st.empty()
    refresh_ai_panel_view()


def render_run_list(runs: list[dict[str, Any]]) -> str:
    """Render run list and return selected run id."""
    st.subheader("Run Selector")
    st.caption("这里用于切换当前查看和追加实验的 run。选择 `All Runs` 时会回到新建 run 视角。")
    clear_disabled = bool(st.session_state.get("ui_locked"))
    if not runs:
        if st.button("Clear All", type="secondary", disabled=clear_disabled):
            ok, response = clear_database_records()
            if ok:
                reset_frontend_state_after_clear()
                st.success(
                    "Database cleared: "
                    f"runs={response.get('deleted_runs', 0)}, "
                    f"experiments={response.get('deleted_experiments', 0)}, "
                    f"results={response.get('deleted_results', 0)}"
                )
                st.rerun()
            else:
                st.error(f'Clear database failed: {response.get("detail", response)}')
        st.info("No runs found. You can still start a new training run from the left panel.")
        st.session_state["selected_run_id"] = "__all__"
        return "__all__"

    run_options = ["__all__"] + [run["id"] for run in runs]
    default_run_id = st.session_state.get("selected_run_id", runs[0]["id"])
    selected_run_index = next((index for index, run_id in enumerate(run_options) if run_id == default_run_id), 0)
    selector_column, clear_column = st.columns([6, 1])
    with selector_column:
        selected_run_id = st.selectbox(
            "Select Run",
            options=run_options,
            index=selected_run_index,
            format_func=lambda run_id: "All Runs" if run_id == "__all__" else next(run["name"] for run in runs if run["id"] == run_id),
        )
    with clear_column:
        st.caption(" ")
        if selected_run_id != "__all__":
            if st.button("Clear", type="secondary", disabled=clear_disabled):
                ok, response = clear_selected_run_records(selected_run_id)
                if ok:
                    reset_frontend_state_after_clear()
                    st.success(
                        "Run cleared: "
                        f"experiments={response.get('deleted_experiments', 0)}, "
                        f"results={response.get('deleted_results', 0)}"
                    )
                    st.rerun()
                else:
                    st.error(f'Clear run failed: {response.get("detail", response)}')

    st.session_state["selected_run_id"] = selected_run_id
    return selected_run_id


def render_training_records_workspace(runs: list[dict[str, Any]], selected_run_id: str) -> None:
    """Render a single-table training records workspace."""
    st.subheader("Training Records")
    st.caption("这里汇总当前范围内的实验记录。勾选记录后可以参与趋势图对比，并在右侧查看详情。")
    all_rows = build_all_training_records(runs, selected_run_id)
    if not all_rows:
        st.info("No training records yet.")
        return

    best_experiment_ids: set[str] = set()
    baseline_experiment_ids: set[str] = set()
    frontier_experiment_ids: set[str] = set()
    if selected_run_id == "__all__":
        for run in runs:
            summary = load_run_summary(run["id"])
            if summary.get("best_experiment_id"):
                best_experiment_ids.add(summary["best_experiment_id"])
            if summary.get("baseline_experiment_id"):
                baseline_experiment_ids.add(summary["baseline_experiment_id"])
            if summary.get("frontier_experiment_id"):
                frontier_experiment_ids.add(summary["frontier_experiment_id"])
    else:
        summary = load_run_summary(selected_run_id)
        if summary.get("best_experiment_id"):
            best_experiment_ids.add(summary["best_experiment_id"])
        if summary.get("baseline_experiment_id"):
            baseline_experiment_ids.add(summary["baseline_experiment_id"])
        if summary.get("frontier_experiment_id"):
            frontier_experiment_ids.add(summary["frontier_experiment_id"])

    selection_key = "training_records_selected_experiments"
    selected_experiment_ids = st.session_state.get(selection_key, [row["experiment_id"] for row in all_rows])
    editor_rows = []
    for row in all_rows:
        editor_row = dict(row)
        editor_row["selected"] = row["experiment_id"] in selected_experiment_ids
        editor_row["best"] = "最佳" if row["experiment_id"] in best_experiment_ids else ""
        if row["experiment_id"] in baseline_experiment_ids:
            editor_row["anchor"] = f"{editor_row.get('anchor', '')}/baseline".strip("/")
        if row["experiment_id"] in frontier_experiment_ids:
            editor_row["anchor"] = f"{editor_row.get('anchor', '')}/frontier".strip("/")
        editor_rows.append(editor_row)

    edited_rows = st.data_editor(
        editor_rows,
        use_container_width=True,
        hide_index=True,
        key="training_records_editor",
        column_config={
            "selected": st.column_config.CheckboxColumn("Compare", help="Include this experiment in the chart comparison"),
            "exp": st.column_config.TextColumn("Exp"),
            "experiment_id": None,
            "best": st.column_config.TextColumn("最佳"),
            "anchor": st.column_config.TextColumn("锚点"),
            "decision": st.column_config.TextColumn("决策"),
        },
        disabled=[
            "exp",
            "best",
            "anchor",
            "decision",
            "run_id",
            "run_name",
            "experiment_id",
            "status",
            "model_name",
            "dataset",
            "top1_acc",
            "val_loss",
            "train_loss",
            "best_epoch",
            "optimizer",
            "learning_rate",
            "batch_size",
            "image_size",
            "epochs",
            "scheduler",
        ],
    )
    selected_experiment_ids = [row["experiment_id"] for row in edited_rows if row["selected"]]
    st.session_state[selection_key] = selected_experiment_ids
    if not selected_experiment_ids:
        st.warning("Select at least one training record to compare in the chart.")
        return

    if selected_run_id != "__all__":
        st.markdown("**Metrics Trend**")
        metric_seed = load_run_metrics(selected_run_id, "top1_acc")
        available_metrics = metric_seed.get("available_metrics", ["top1_acc", "val_loss", "train_loss"])
        metric_name = st.selectbox("Metric", available_metrics)
        metrics = metric_seed if metric_name == metric_seed.get("metric_name") else load_run_metrics(selected_run_id, metric_name)
        filtered_points = [
            point for point in metrics["points"] if point["experiment_id"] in selected_experiment_ids
        ]
        if filtered_points:
            st.line_chart(
                [
                    {
                        "experiment_index": point["experiment_index"],
                        metric_name: point["metric_value"],
                    }
                    for point in filtered_points
                ],
                x="experiment_index",
                y=metric_name,
                use_container_width=True,
            )
        else:
            st.info("The selected training records do not have visible metric points yet.")
    else:
        st.caption("Metrics trend is shown only when a single run is selected.")


def render_result_workspace(runs: list[dict[str, Any]], selected_run_id: str) -> None:
    """Render the right-side result workspace."""
    st.subheader("Results")
    st.caption("这里展示当前选中 run 或实验的结果、指标、参数快照和搜索策略。")
    selected_experiment_id = st.session_state.get("selected_experiment_id")
    best_experiment_detail = None
    run_summary = None
    if selected_run_id not in {"", "__all__"}:
        run_summary = load_run_summary(selected_run_id)
        best_experiment_detail = get_best_experiment_detail(selected_run_id)
        if not selected_experiment_id and best_experiment_detail is not None:
            selected_experiment_id = best_experiment_detail["id"]
            st.session_state["selected_experiment_id"] = selected_experiment_id

    inspected_experiment_detail = load_experiment_detail(selected_experiment_id) if selected_experiment_id else None
    summary_experiment_detail = best_experiment_detail or inspected_experiment_detail

    if summary_experiment_detail:
        result = summary_experiment_detail.get("result") or {}
        metrics = result.get("metrics") or {}
        metric_columns = st.columns(4)
        metric_columns[0].metric("Status", summary_experiment_detail.get("status", "-"))
        metric_columns[1].metric("Top1 Acc", metrics.get("top1_acc", "-"))
        metric_columns[2].metric("Val Loss", metrics.get("val_loss", "-"))
        metric_columns[3].metric("Train Loss", metrics.get("train_loss", "-"))
        if run_summary is not None:
            st.caption(
                " / ".join(
                    [
                        f"baseline={get_short_experiment_id(run_summary.get('baseline_experiment_id'))}",
                        f"best={get_short_experiment_id(run_summary.get('best_experiment_id'))}",
                        f"frontier={get_short_experiment_id(run_summary.get('frontier_experiment_id'))}",
                    ]
                )
            )

        config = summary_experiment_detail.get("config") or {}
        search_policy = config.get("search_policy") or {}
        ranking_policy = config.get("ranking_policy") or {}
        if search_policy:
            st.caption(
                "AI search: "
                + ", ".join(
                    [
                        f"basic_hparams={search_policy.get('allow_basic_hparam_search')}",
                        f"strategy={search_policy.get('allow_strategy_search')}",
                        f"loss={search_policy.get('allow_loss_search')}",
                        f"augmentation={search_policy.get('allow_augmentation_search')}",
                        f"manual_approval={search_policy.get('require_manual_approval_for_high_impact_changes')}",
                    ]
                )
            )
        if ranking_policy:
            st.caption("Ranking policy: " + summarize_ranking_policy(ranking_policy))

        with st.container(border=True):
            title = "Best Experiment Detail" if best_experiment_detail is not None else "Experiment Detail"
            st.markdown(f"**{title}**")
            st.json(
                {
                    "experiment_id": summary_experiment_detail.get("id"),
                    "run_id": summary_experiment_detail.get("run_id"),
                    "decision": summary_experiment_detail.get("decision"),
                    "decision_reason": summary_experiment_detail.get("decision_reason"),
                    "baseline_experiment_id": summary_experiment_detail.get("baseline_experiment_id"),
                    "is_best_so_far": summary_experiment_detail.get("is_best_so_far"),
                    "config": summary_experiment_detail.get("config"),
                    "result": summary_experiment_detail.get("result"),
                    "reflection": summary_experiment_detail.get("reflection"),
                },
                expanded=False,
            )
    else:
        st.caption("Select or create a run to inspect results.")

    render_ai_suggestion_panel()
    render_training_records_workspace(runs, selected_run_id)


def main() -> None:
    """Render the Streamlit demo UI."""
    st.set_page_config(page_title="AutoVisionLab Demo", layout="wide")
    st.title("AutoVisionLab Demo")
    st.caption("左侧训练，右侧看结果。`Train` 单次执行后给建议；`Auto Train` 自动连续调优并总结本轮优化结果。")
    post_action_notice = st.session_state.pop("post_action_notice", None)
    if post_action_notice:
        if post_action_notice["level"] == "success":
            st.success(post_action_notice["message"])
        else:
            st.error(post_action_notice["message"])

    sync_auto_train_task_state()

    runs = load_runs()
    is_training_active = has_running_experiment(runs) or bool(st.session_state.get("ui_locked"))
    if is_training_active and st.session_state.get("active_train_control") is None:
        st.session_state["active_train_control"] = "manual"
    left_column, right_column = st.columns([1, 1.45], gap="large")

    with left_column:
        selected_run_id = render_run_list(runs)
        render_control_panel(runs, selected_run_id, is_training_active)
        render_activity_log()

    with right_column:
        render_result_workspace(runs, selected_run_id)

    render_live_training_monitor()
    process_pending_train_request()

    training_experiment_id = st.session_state.get("training_experiment_id")
    current_auto_task_id = st.session_state.get("current_auto_task_id")
    active_train_control = st.session_state.get("active_train_control")
    if training_experiment_id and not current_auto_task_id and active_train_control != "auto":
        training_experiment_detail = load_experiment_detail(training_experiment_id)
        training_status = training_experiment_detail.get("status")
        if training_status == "running":
            if st.session_state.get("last_running_experiment_id") != training_experiment_id:
                append_activity_log(f"Experiment {training_experiment_id} is running.")
                st.session_state["last_running_experiment_id"] = training_experiment_id
        elif st.session_state.get("last_finished_experiment_id") != training_experiment_id:
            st.session_state["last_finished_experiment_id"] = training_experiment_id
            st.session_state["selected_experiment_id"] = training_experiment_id
            clear_training_state()
            if training_status == "discarded":
                append_activity_log(f"Experiment {training_experiment_id} was discarded.")
                st.warning(f"Training stopped and discarded: {training_experiment_id}")
                return
            append_activity_log(
                f"Experiment {training_experiment_id} finished with status {training_status} "
                f"and {format_metric_summary(training_experiment_detail)}."
            )
            completed_run_id = training_experiment_detail.get("run_id")
            if completed_run_id:
                ok, suggestion_message = generate_and_store_ai_suggestion(
                    completed_run_id,
                    training_experiment_id,
                    training_experiment_detail,
                    "Post-train",
                )
                if not ok:
                    append_activity_log(suggestion_message)
            st.success(f"Training finished with status: {training_status}")


if __name__ == "__main__":
    main()
