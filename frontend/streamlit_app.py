"""Streamlit demo UI for the autonomous training platform."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
import time
from typing import Any

import altair as alt
import pandas as pd
import requests
import streamlit as st

from manual_train_logic import resolve_manual_train_action


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
    "mobilenet_v3_small": {
        "model_family": "mobilenet",
        "parameter_space_version": "mobilenet_v3_small@v1",
        "parameter_space": {
            "model_name": "mobilenet_v3_small",
            "version": "mobilenet_v3_small@v1",
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
                "neck_name": {"type": "enum", "choices": ["avg_pool", "gem_pool"]},
                "head_name": {"type": "enum", "choices": ["native_classifier", "linear", "dropout_linear"]},
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
}
MODEL_LABELS = {
    "mobilenet_v2": "MobileNetV2",
    "mobilenet_v3_small": "MobileNetV3 Small",
    "googlenet": "GoogLeNet",
    "resnet18": "ResNet18",
}
FALLBACK_DATASET_CATALOG = [
    {
        "name": "cifar10",
        "is_ready_for_training": True,
        "classification_dir": "data/classification/cifar10",
        "original_image_size": 32,
        "image_size_options": [32, 64, 96],
        "message": "Fallback dataset entry.",
    },
    {
        "name": "neu",
        "is_ready_for_training": True,
        "classification_dir": "data/classification/neu",
        "original_image_size": 200,
        "image_size_options": [200, 224, 256],
        "message": "Fallback dataset entry.",
    },
]

LOG_LIMIT = 60
LIVE_LOG_CONTAINER: Any | None = None
LIVE_AI_PANEL_CONTAINER: Any | None = None
REQUEST_CACHE_TTL_SECONDS = 0.2
REQUEST_CACHE: dict[str, tuple[float, Any]] = {}
AI_BLOCKED_CHANGE_FIELDS = {"epochs"}
RANKING_METRIC_LABELS = {
    "top1_acc": "Top1 Acc",
    "val_loss": "Val Loss",
    "training_seconds": "Training Seconds",
    "latency_ms": "Latency (ms)",
    "parameter_count_million": "Params (M)",
}
WORKSPACE_VIEW_HOME = "home"
WORKSPACE_VIEW_SINGLE_MODEL = "single_model_workspace"
WORKSPACE_VIEW_COMPARE = "compare_workspace"


def default_search_policy() -> dict[str, Any]:
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
        "allow_model_module_search": False,
        "require_manual_approval_for_high_impact_changes": True,
    }


def default_ranking_policy() -> dict[str, Any]:
    """Return the default ranking policy for the UI."""
    return {
        "primary_metric": "top1_acc",
        "primary_metric_mode": "max",
        "min_primary_metric_improvement": 0.01,
        "primary_metric_parity_epsilon": 0.0005,
        "tie_breaker_metric": "latency_ms",
        "tie_breaker_mode": "min",
        "min_tie_breaker_metric_improvement": 0.5,
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
    dataset_label = get_dataset_run_name_label(dataset)
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
    dataset_info = get_dataset_info(dataset)
    original_image_size = dataset_info.get("original_image_size")
    if isinstance(original_image_size, int) and original_image_size > 0:
        return original_image_size
    return 64


def get_dataset_image_size_options(dataset: str) -> list[int]:
    """Return allowed image size options for one dataset."""
    dataset_info = get_dataset_info(dataset)
    image_size_options = dataset_info.get("image_size_options") or []
    if image_size_options:
        return [int(size) for size in image_size_options]
    return [get_original_image_size(dataset, "")]


def load_dataset_catalog() -> list[dict[str, Any]]:
    """Load dataset readiness info from the backend."""
    payload = request_json("/datasets", FALLBACK_DATASET_CATALOG)
    return payload if isinstance(payload, list) and payload else deepcopy(FALLBACK_DATASET_CATALOG)


def get_dataset_catalog() -> list[dict[str, Any]]:
    """Return the active dataset catalog for the current session."""
    catalog = st.session_state.get("dataset_catalog")
    if isinstance(catalog, list) and catalog:
        return catalog
    return deepcopy(FALLBACK_DATASET_CATALOG)


def get_dataset_info(dataset_name: str) -> dict[str, Any]:
    """Return one dataset entry from the current catalog."""
    for dataset_info in get_dataset_catalog():
        if dataset_info.get("name") == dataset_name:
            return dataset_info
    for dataset_info in FALLBACK_DATASET_CATALOG:
        if dataset_info.get("name") == dataset_name:
            return dataset_info
    return {"name": dataset_name, "original_image_size": 64, "image_size_options": [64]}


def get_dataset_run_name_label(dataset_name: str) -> str:
    """Return the label used for auto-generated run names."""
    return str(get_dataset_info(dataset_name).get("name") or dataset_name)


def is_oom_failure(training_experiment_detail: dict[str, Any]) -> bool:
    """Return whether one failed experiment looks like a CUDA OOM."""
    decision_reason = str(training_experiment_detail.get("decision_reason") or "")
    return "cuda out of memory" in decision_reason.lower()


def get_ready_dataset_options(current_dataset: str | None = None) -> list[str]:
    """Return dataset names that should appear in the dataset selectbox."""
    visible_datasets = [
        dataset_info["name"]
        for dataset_info in get_dataset_catalog()
        if (
            dataset_info.get("classification_dir")
            and dataset_info.get("train_manifest_exists")
            and dataset_info.get("val_manifest_exists")
        )
    ]
    if not visible_datasets:
        visible_datasets = [
            dataset_info["name"]
            for dataset_info in FALLBACK_DATASET_CATALOG
            if dataset_info.get("classification_dir")
        ]
    if current_dataset and current_dataset not in visible_datasets:
        visible_datasets.append(current_dataset)
    return visible_datasets


def get_default_dataset_name() -> str:
    """Return the preferred default dataset for the UI."""
    ready_dataset_options = get_ready_dataset_options()
    if "neu" in ready_dataset_options:
        return "neu"
    if ready_dataset_options:
        return ready_dataset_options[0]
    return "cifar10"


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


def format_metric_value(metric_name: str, metric_value: Any) -> str:
    """Format one scalar metric value for compact display."""
    if not isinstance(metric_value, (int, float)):
        return "-"
    if metric_name in {"top1_acc", "val_loss", "train_loss"}:
        return f"{float(metric_value):.4f}"
    if metric_name == "latency_ms":
        return f"{float(metric_value):.3f}"
    if metric_name == "parameter_count_million":
        return f"{float(metric_value):.3f}"
    if metric_name == "training_seconds":
        return format_elapsed_seconds(float(metric_value))
    if float(metric_value).is_integer():
        return str(int(metric_value))
    return f"{float(metric_value):.3f}"


def format_token_estimate(token_count: int | None) -> str:
    """Format one rough token estimate for compact UI display."""
    if not isinstance(token_count, int) or token_count <= 0:
        return "-"
    if token_count >= 10000:
        trimmed = f"{token_count / 10000:.1f}".rstrip("0").rstrip(".")
        return f"{trimmed}w"
    return f"{token_count}"


def format_token_breakdown(
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
) -> str | None:
    """Format one provider token usage triplet for compact UI display."""
    if not any(
        isinstance(token_count, int) and token_count > 0
        for token_count in (prompt_tokens, completion_tokens, total_tokens)
    ):
        return None
    prompt_label = format_token_estimate(prompt_tokens)
    completion_label = format_token_estimate(completion_tokens)
    total_label = format_token_estimate(total_tokens)
    return f"input {prompt_label} / output {completion_label} / total {total_label}"


def is_network_related_auto_train_error(error_message: str | None) -> bool:
    """Return whether one auto-train task error looks like a network/provider failure."""
    if not error_message:
        return False
    normalized_message = error_message.lower()
    network_markers = (
        "httpsconnectionpool",
        "max retries exceeded",
        "sslerror",
        "ssleoferror",
        "unexpected_eof_while_reading",
        "connection aborted",
        "connection reset",
        "read timed out",
        "connect timeout",
        "temporary failure in name resolution",
    )
    return any(marker in normalized_message for marker in network_markers)


def build_auto_train_terminal_message(
    task_status: str | None,
    *,
    stop_reason: str | None = None,
    task_error: str | None = None,
) -> str:
    """Build one user-facing terminal status message for model search."""
    if task_status == "stopped_by_policy":
        return stop_reason or "Model Search 已按停止策略结束。"
    if task_status == "stopped":
        return stop_reason or "Model Search 已手动停止，当前实验已丢弃。"
    if task_status == "failed":
        if is_network_related_auto_train_error(task_error):
            return (
                "Model Search 异常结束：生成下一轮 AI proposal 时网络或上游模型服务连接异常。"
                f"{f' 原始错误：{task_error}' if task_error else ''}"
            )
        return f"Model Search 异常结束：{task_error}" if task_error else "Model Search 异常结束。"
    return stop_reason or "Model Search 已结束。"


def get_experiment_training_time_label(experiment_detail: dict[str, Any]) -> str:
    """Return the completed training duration label for one experiment."""
    resource = (experiment_detail.get("result") or {}).get("resource") or {}
    training_seconds = resource.get("training_seconds")
    if isinstance(training_seconds, (int, float)) and training_seconds >= 0:
        return format_elapsed_seconds(float(training_seconds))
    return "-"


def get_running_elapsed_seconds(run_id: str, experiment_id: str) -> float | None:
    """Return the live elapsed seconds for the currently running experiment when available."""
    auto_task_progress = st.session_state.get("auto_task_progress") or {}
    if (
        st.session_state.get("current_auto_task_id")
        and auto_task_progress.get("current_experiment_id") == experiment_id
        and st.session_state.get("selected_run_id") == run_id
    ):
        elapsed_seconds = auto_task_progress.get("elapsed_seconds")
        return float(elapsed_seconds) if isinstance(elapsed_seconds, (int, float)) else None

    started_at = st.session_state.get("training_started_at_timestamp")
    active_experiment_id = st.session_state.get("training_experiment_id")
    if active_experiment_id == experiment_id and isinstance(started_at, (int, float)):
        return max(0.0, time.time() - float(started_at))
    return None


def build_running_result_placeholder(
    run_id: str,
    status: str,
    *,
    experiment_id: str | None = None,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    """Build a lightweight running result card payload when the task has started but no finished result exists yet."""
    return {
        "id": experiment_id or "pending",
        "run_id": run_id,
        "status": status,
        "result": {"metrics": {}},
        "_elapsed_seconds_override": elapsed_seconds,
    }


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
    allow_model_module_search: bool,
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
    if allow_model_module_search:
        field_labels.extend(["neck_name", "head_name"])

    deduped_labels = list(dict.fromkeys(field_labels))
    if not deduped_labels:
        return "当前 AI 不会自动搜索任何参数。"
    return "当前 AI 会搜索： " + ", ".join(deduped_labels)


def request_json(path: str, fallback: Any) -> Any:
    """Fetch JSON from the backend and fall back to local demo data."""
    cached_entry = REQUEST_CACHE.get(path)
    if cached_entry is not None:
        cached_at, cached_payload = cached_entry
        if time.monotonic() - cached_at <= REQUEST_CACHE_TTL_SECONDS:
            return deepcopy(cached_payload)
    try:
        response = requests.get(f"{API_BASE_URL}{path}", timeout=0.8)
        response.raise_for_status()
        payload = unwrap_api_response(response.json())
        REQUEST_CACHE[path] = (time.monotonic(), deepcopy(payload))
        return payload
    except requests.RequestException:
        REQUEST_CACHE[path] = (time.monotonic(), deepcopy(fallback))
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
) -> tuple[bool, Any]:
    """Start one backend auto-train task."""
    request_payload = {
        "run_id": None if selected_run_id == "__all__" else selected_run_id,
        "run_name": payload["name"],
        "dataset": payload["dataset"],
        "model_name": payload["model_name"],
        "config": payload["config"],
        "parameter_space": payload["parameter_space"],
    }
    return post_json("/runs/auto-train", request_payload)


def start_model_compare_task_request(payload: dict[str, Any]) -> tuple[bool, Any]:
    """Start one backend cross-model compare task."""
    request_payload = {
        "dataset": payload["dataset"],
        "config": payload["config"],
    }
    return post_json("/runs/model-compare", request_payload)


def load_auto_train_task(task_id: str) -> tuple[bool, Any]:
    """Load one backend auto-train task."""
    payload = request_json(f"/runs/auto-train/{task_id}", {"detail": "Auto train task not found"})
    return isinstance(payload, dict) and "task_id" in payload, payload


def load_active_auto_train_task() -> tuple[bool, Any]:
    """Load the currently active auto-train task when one exists."""
    payload = request_json("/runs/auto-train/active", {"detail": "No active auto train task"})
    return isinstance(payload, dict) and "task_id" in payload, payload


def load_model_compare_task(task_id: str) -> tuple[bool, Any]:
    """Load one backend cross-model compare task."""
    payload = request_json(f"/runs/model-compare/{task_id}", {"detail": "Model compare task not found"})
    return isinstance(payload, dict) and "task_id" in payload, payload


def load_active_model_compare_task() -> tuple[bool, Any]:
    """Load the currently active cross-model compare task when one exists."""
    payload = request_json("/runs/model-compare/active", {"detail": "No active model compare task"})
    return isinstance(payload, dict) and "task_id" in payload, payload


def stop_auto_train_task_request(task_id: str) -> tuple[bool, Any]:
    """Stop one backend auto-train task."""
    return post_without_body(f"/runs/auto-train/{task_id}/stop")


def start_experiment_suggestion_task_request(experiment_id: str) -> tuple[bool, Any]:
    """Start one background suggestion task for a completed experiment."""
    return post_without_body(f"/experiments/{experiment_id}/suggestion")


def load_experiment_suggestion_task(experiment_id: str) -> tuple[bool, Any]:
    """Load the latest suggestion task for one experiment."""
    payload = request_json(
        f"/experiments/{experiment_id}/suggestion",
        {"detail": "Experiment suggestion task not found"},
    )
    return isinstance(payload, dict) and "task_id" in payload, payload


def load_runs() -> list[dict[str, Any]]:
    """Return all runs."""
    return request_json(
        "/runs",
        [
            {
                "id": "run_demo_001",
                "name": "CIFAR-10 MobileNet baseline tuning",
                "dataset": "cifar10",
                "model_name": "mobilenet_v3_small",
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
            "model_name": "mobilenet_v3_small",
            "status": "active",
            "notes": "Using Streamlit fallback demo data because backend is not reachable.",
            "baseline_experiment_id": "exp_demo_001",
            "best_experiment_id": "exp_demo_002",
            "frontier_experiment_id": "exp_demo_002",
            "experiments": [
                {"id": "exp_demo_001", "run_id": run_id, "status": "success", "model_name": "mobilenet_v3_small", "decision": "keep", "is_best_so_far": False},
                {"id": "exp_demo_002", "run_id": run_id, "status": "success", "model_name": "mobilenet_v3_small", "decision": "keep", "is_best_so_far": True},
                {"id": "exp_demo_003", "run_id": run_id, "status": "success", "model_name": "mobilenet_v3_small", "decision": "discard", "is_best_so_far": False},
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
                "model_name": "mobilenet_v3_small",
                "parameter_space_version": "mobilenet_v3_small@v1",
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
                "model_name": "mobilenet_v3_small",
                "version": "mobilenet_v3_small@v1",
                "editable_params": {},
            },
            "proposal": {
                "task_type": "classification",
                "model_name": "mobilenet_v3_small",
                "based_on_experiment_ids": ["exp_demo_001"],
                "hypothesis": "Add light mixup and slightly lower label smoothing for the next round.",
                "changes": {"mixup_alpha": 0.2, "label_smoothing": 0.05},
                "train_hyp_changes": {
                    "augmentation": {"mixup": 0.2},
                    "label_smoothing": 0.05
                },
                "recipe_changes": None,
                "reason": "The previous baseline is stable enough to try one augmentation change plus one nearby regularization adjustment.",
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
    visible_log_lines: list[str] = []
    for log_line in log_lines:
        normalized_line = log_line.split("] ", 1)[-1]
        if "AI suggestion:" in normalized_line or "Proposed changes:" in normalized_line:
            continue
        if normalized_line.startswith("Round ") and "AI suggested" in normalized_line:
            continue
        if normalized_line.startswith("Round ") and "top1_acc=" in normalized_line:
            continue
        if normalized_line.startswith("Baseline finished:"):
            continue
        visible_log_lines.append(log_line)
    log_text = "\n".join(visible_log_lines[-20:]) if visible_log_lines else "Logs will appear here once training starts."
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


def format_structured_changes(changes: dict[str, Any] | None) -> str:
    """Format one nested recipe-oriented change payload for display."""
    if not changes:
        return "-"
    return json.dumps(changes, ensure_ascii=False, sort_keys=True)


def _compare_top1_acc_key(candidate: dict[str, Any]) -> float:
    """Return a sortable top1_acc key for one compare candidate."""
    metric = candidate.get("top1_acc")
    return float(metric) if isinstance(metric, (int, float)) else float("-inf")


def _compare_latency_key(candidate: dict[str, Any]) -> float:
    """Return a sortable latency key for one compare candidate."""
    metric = candidate.get("latency_ms")
    return float(metric) if isinstance(metric, (int, float)) else float("inf")


def _compare_parameter_key(candidate: dict[str, Any]) -> float:
    """Return a sortable parameter count key for one compare candidate."""
    metric = candidate.get("parameter_count_million")
    return float(metric) if isinstance(metric, (int, float)) else float("inf")


def build_model_compare_ai_panel(
    compare_summary: dict[str, Any] | None,
    *,
    task_status: str | None,
    task_error: str | None = None,
    progress: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one compare-stage summary payload for the AI panel."""
    candidate_results = list((compare_summary or {}).get("candidate_results") or [])
    successful_candidates = [
        candidate
        for candidate in candidate_results
        if candidate.get("status") == "success"
    ]
    progress = progress or {}
    if task_status in {"queued", "running"}:
        current_model_name = progress.get("current_model_name") or "-"
        current_model_index = int(progress.get("current_model_index") or 0)
        total_models = int(progress.get("total_models") or 0)
        elapsed_seconds = float(progress.get("elapsed_seconds") or 0.0)
        return {
            "mode": "compare",
            "status": task_status,
            "headline": "正在执行跨模型 shared baseline 比较。",
            "summary_lines": [
                f"当前模型：{get_model_label(str(current_model_name))}",
                f"进度：{current_model_index}/{total_models}",
                f"已运行：{format_elapsed_seconds(elapsed_seconds)}",
            ],
            "candidate_results": candidate_results,
        }

    if task_status == "failed":
        return {
            "mode": "compare",
            "status": task_status,
            "headline": "Models Compare 未成功完成。",
            "summary_lines": [task_error or "All compare candidates failed."],
            "candidate_results": candidate_results,
        }

    if not successful_candidates:
        return {
            "mode": "compare",
            "status": task_status or "success",
            "headline": "Models Compare 已结束，但没有可继续的成功候选。",
            "summary_lines": ["没有成功完成 baseline 的候选模型。"],
            "candidate_results": candidate_results,
        }

    ranked_candidates = sorted(
        successful_candidates,
        key=lambda candidate: (
            -_compare_top1_acc_key(candidate),
            _compare_latency_key(candidate),
            _compare_parameter_key(candidate),
            str(candidate.get("model_name") or ""),
        ),
    )
    recommended_candidate = ranked_candidates[0]
    best_accuracy_candidate = max(successful_candidates, key=_compare_top1_acc_key)
    fastest_candidate = min(successful_candidates, key=_compare_latency_key)

    recommendation_reasons = [
        (
            f"{get_model_label(str(recommended_candidate.get('model_name')))} "
            f"当前在 compare 结果里综合排名最高。"
        )
    ]
    if recommended_candidate.get("model_name") == best_accuracy_candidate.get("model_name"):
        recommendation_reasons.append("它拿到了当前最高的 top1_acc。")
    else:
        recommendation_reasons.append(
            (
                f"虽然最高精度来自 {get_model_label(str(best_accuracy_candidate.get('model_name')))}，"
                f"但推荐模型的精度更接近且延迟/规模更均衡。"
            )
        )
    if recommended_candidate.get("model_name") == fastest_candidate.get("model_name"):
        recommendation_reasons.append("它同时也是当前延迟最小的候选。")
    else:
        recommendation_reasons.append(
            (
                f"最快的是 {get_model_label(str(fastest_candidate.get('model_name')))}，"
                "但当前推荐更适合作为下一阶段继续优化的起点。"
            )
        )

    leaderboard_lines = [
        (
            f"{index}. {get_model_label(str(candidate.get('model_name')))} | "
            f"acc={format_metric_value('top1_acc', candidate.get('top1_acc'))} | "
            f"latency={format_metric_value('latency_ms', candidate.get('latency_ms'))} ms | "
            f"params={format_metric_value('parameter_count_million', candidate.get('parameter_count_million'))} M"
        )
        for index, candidate in enumerate(ranked_candidates, start=1)
    ]
    return {
        "mode": "compare",
        "status": task_status or "success",
        "headline": "Models Compare 已完成。",
        "summary_lines": [
            f"成功候选：{len(successful_candidates)}/{len(candidate_results)}",
            (
                f"最高精度：{get_model_label(str(best_accuracy_candidate.get('model_name')))} "
                f"({format_metric_value('top1_acc', best_accuracy_candidate.get('top1_acc'))})"
            ),
            (
                f"最低延迟：{get_model_label(str(fastest_candidate.get('model_name')))} "
                f"({format_metric_value('latency_ms', fastest_candidate.get('latency_ms'))} ms)"
            ),
        ],
        "recommended_candidate": recommended_candidate,
        "recommendation_reasons": recommendation_reasons,
        "leaderboard_lines": leaderboard_lines,
        "candidate_results": candidate_results,
    }


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
    elif active_mode == "compare":
        suggestion_payload = st.session_state.get("compare_ai_panel")
    elif active_mode == "manual":
        suggestion_payload = st.session_state.get("manual_ai_panel")
    else:
        suggestion_payload = (
            st.session_state.get("auto_train_summary")
            or st.session_state.get("compare_ai_panel")
            or st.session_state.get("manual_ai_panel")
        )
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
                task_error = suggestion_payload.get("task_error")
                progress = st.session_state.get("auto_task_progress") or {}
                st.markdown("**Model Search Summary**")
                st.markdown("**Summary**")
                if progress:
                    completed_rounds = len(rounds)
                    current_round = progress.get("current_round", 0)
                    elapsed_seconds = progress.get("elapsed_seconds", 0.0)
                    current_experiment_id = progress.get("current_experiment_id") or "-"
                    latest_prompt_tokens_estimate = progress.get("latest_prompt_tokens_estimate")
                    estimated_prompt_tokens_total = progress.get("estimated_prompt_tokens_total")
                    latest_prompt_history_items = progress.get("latest_prompt_history_items")
                    latest_provider_usage = format_token_breakdown(
                        progress.get("latest_provider_prompt_tokens"),
                        progress.get("latest_provider_completion_tokens"),
                        progress.get("latest_provider_total_tokens"),
                    )
                    cumulative_provider_usage = format_token_breakdown(
                        progress.get("provider_prompt_tokens_total"),
                        progress.get("provider_completion_tokens_total"),
                        progress.get("provider_total_tokens_total"),
                    )
                    progress_parts = [
                        f"进度：已完成 {completed_rounds} 轮",
                        f"当前轮次 {current_round}",
                        f"已运行 {format_elapsed_seconds(elapsed_seconds)}",
                        f"实验 {get_short_experiment_id(current_experiment_id)}",
                    ]
                    if latest_provider_usage:
                        progress_parts.append(f"本轮 tokens {latest_provider_usage}")
                    elif isinstance(latest_prompt_tokens_estimate, int) and latest_prompt_tokens_estimate > 0:
                        progress_parts.append(
                            f"本轮 proposal 约 {format_token_estimate(latest_prompt_tokens_estimate)} tokens"
                        )
                    if cumulative_provider_usage:
                        progress_parts.append(f"累计 tokens {cumulative_provider_usage}")
                    elif isinstance(estimated_prompt_tokens_total, int) and estimated_prompt_tokens_total > 0:
                        progress_parts.append(
                            f"累计约 {format_token_estimate(estimated_prompt_tokens_total)} tokens"
                        )
                    if isinstance(latest_prompt_history_items, int) and latest_prompt_history_items > 0:
                        progress_parts.append(f"history {latest_prompt_history_items} 条")
                    st.caption(
                        "，".join(progress_parts) + "。"
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
                    trend_frame = pd.DataFrame(trend_rows).set_index("round_index")
                    st.line_chart(trend_frame, use_container_width=True)
                if rounds:
                    keep_count = sum(1 for round_info in rounds if round_info["result"].get("decision") == "keep")
                    discard_count = sum(1 for round_info in rounds if round_info["result"].get("decision") == "discard")
                    with st.expander(
                        f"Round History ({len(rounds)} rounds, {keep_count} keep, {discard_count} discard)",
                        expanded=False,
                    ):
                        for round_info in reversed(rounds[-5:]):
                            result_snapshot = round_info["result"]
                            round_label = (
                                f"第 {round_info['round_index']} 轮 | "
                                f"{format_proposal_changes(round_info['proposal']['changes'])} | "
                                f"{result_snapshot.get('decision') or result_snapshot.get('status')}"
                            )
                            with st.expander(round_label, expanded=False):
                                st.caption(
                                    f"实验 {get_short_experiment_id(result_snapshot.get('experiment_id'))} | "
                                    f"{result_snapshot.get('summary')}"
                                )
                                st.markdown(f"**Hypothesis**  \n{round_info['proposal'].get('hypothesis', '-')}")
                                st.caption(round_info["proposal"].get("reason", "-"))
                                if round_info["proposal"].get("train_hyp_changes"):
                                    st.markdown(
                                        f"**Train Hyp Changes**  \n`{format_structured_changes(round_info['proposal'].get('train_hyp_changes'))}`"
                                    )
                                if round_info["proposal"].get("recipe_changes"):
                                    st.markdown(
                                        f"**Recipe Changes**  \n`{format_structured_changes(round_info['proposal'].get('recipe_changes'))}`"
                                    )
                                st.markdown(
                                    f"**Based On**  \n{', '.join(round_info['proposal'].get('based_on_experiment_ids') or []) or '-'}"
                                )
                                st.markdown(
                                    f"**Decision Reason**  \n{result_snapshot.get('decision_reason') or '-'}"
                                )
                if task_status in {"stopped", "stopped_by_policy", "failed"}:
                    st.markdown("**Why It Stopped**")
                    st.caption(
                        build_auto_train_terminal_message(
                            task_status,
                            stop_reason=stop_reason,
                            task_error=task_error,
                        )
                    )
                if final_proposal:
                    st.markdown("**Next Suggestion**")
                    st.markdown(final_proposal["hypothesis"])
                    st.caption(final_proposal["reason"])
                    st.markdown(f"`{format_proposal_changes(final_proposal['changes'])}`")
                    if final_proposal.get("train_hyp_changes"):
                        st.markdown(
                            f"**Train Hyp Changes**  \n`{format_structured_changes(final_proposal.get('train_hyp_changes'))}`"
                        )
                    if final_proposal.get("recipe_changes"):
                        st.markdown(
                            f"**Recipe Changes**  \n`{format_structured_changes(final_proposal.get('recipe_changes'))}`"
                        )
                elif task_status in {"stopped", "stopped_by_policy", "failed"}:
                    final_suggestion_error = suggestion_payload.get("final_suggestion_error")
                    if final_suggestion_error:
                        st.caption(f"Next suggestion is not available: {final_suggestion_error}")
                else:
                    st.caption("Model Search 进行中。每完成一轮后，这里的趋势会自动更新。")
                return

            if suggestion_payload.get("mode") == "compare":
                st.markdown("**Models Compare Summary**")
                st.caption(suggestion_payload.get("headline", ""))
                for summary_line in suggestion_payload.get("summary_lines") or []:
                    st.caption(summary_line)

                recommended_candidate = suggestion_payload.get("recommended_candidate")
                if recommended_candidate:
                    st.markdown(
                        f"**Recommended Next Model**  \n{get_model_label(str(recommended_candidate.get('model_name')))}"
                    )
                    st.caption(
                        (
                            f"acc={format_metric_value('top1_acc', recommended_candidate.get('top1_acc'))} | "
                            f"latency={format_metric_value('latency_ms', recommended_candidate.get('latency_ms'))} ms | "
                            f"params={format_metric_value('parameter_count_million', recommended_candidate.get('parameter_count_million'))} M"
                        )
                    )
                    recommendation_reasons = suggestion_payload.get("recommendation_reasons") or []
                    if recommendation_reasons:
                        st.markdown("**Why This Model**")
                        for reason in recommendation_reasons:
                            st.caption(reason)

                leaderboard_lines = suggestion_payload.get("leaderboard_lines") or []
                if leaderboard_lines:
                    st.markdown("**Ranked Snapshot**")
                    for leaderboard_line in leaderboard_lines:
                        st.caption(leaderboard_line)

                if suggestion_payload.get("status") == "success":
                    st.markdown("**Next Step**  \n从 compare 结果里手动选择一个模型，进入单模型优化流程。")
                return

            proposal = suggestion_payload["proposal"]
            result = suggestion_payload["result"]
            st.markdown("**Single-Run Suggestion**")
            st.caption(f"实验 {get_short_experiment_id(result['experiment_id'])} | {result['summary']}")
            suggestion_status = suggestion_payload.get("suggestion_status")
            suggestion_error = suggestion_payload.get("suggestion_error")
            token_usage = suggestion_payload.get("token_usage") or {}
            token_usage_label = format_token_breakdown(
                token_usage.get("prompt_tokens"),
                token_usage.get("completion_tokens"),
                token_usage.get("total_tokens"),
            )
            if suggestion_status in {"queued", "running"} and not proposal:
                st.caption("AI suggestion is generating in the background.")
                if token_usage_label:
                    st.caption(f"当前 tokens: {token_usage_label}")
                return
            if suggestion_status == "failed" and not proposal:
                st.caption(f"AI suggestion is not available: {suggestion_error or 'unknown error'}")
                return
            st.markdown(f"**Suggestion**  \n{proposal['hypothesis']}")
            st.caption(proposal["reason"])
            if token_usage_label:
                st.caption(f"Token usage: {token_usage_label}")
            st.markdown(f"**Suggested Changes**  \n{format_proposal_changes(proposal['changes'])}")
            if proposal.get("train_hyp_changes"):
                st.markdown(
                    f"**Train Hyp Changes**  \n`{format_structured_changes(proposal.get('train_hyp_changes'))}`"
                )
            if proposal.get("recipe_changes"):
                st.markdown(
                    f"**Recipe Changes**  \n`{format_structured_changes(proposal.get('recipe_changes'))}`"
                )

def set_ui_locked(is_locked: bool) -> None:
    """Lock or unlock all train actions."""
    st.session_state["ui_locked"] = is_locked


def set_post_action_notice(message: str, level: str = "success") -> None:
    """Store one notice to be shown after the next rerun."""
    st.session_state["post_action_notice"] = {"message": message, "level": level}


def set_workspace_view(workspace_view: str) -> None:
    """Switch the top-level frontend workspace."""
    st.session_state["workspace_view"] = workspace_view


def get_workspace_view() -> str:
    """Return the current top-level workspace view."""
    return str(st.session_state.get("workspace_view") or WORKSPACE_VIEW_HOME)


def sync_workspace_view() -> None:
    """Keep the current workspace aligned with active tasks and recent user actions."""
    if st.session_state.get("current_compare_task_id"):
        set_workspace_view(WORKSPACE_VIEW_COMPARE)
        return
    if st.session_state.get("current_auto_task_id"):
        set_workspace_view(WORKSPACE_VIEW_SINGLE_MODEL)
        return

    current_view = get_workspace_view()
    selected_run_id = str(st.session_state.get("selected_run_id") or "__all__")
    compare_summary = st.session_state.get("model_compare_summary") or {}
    if current_view == WORKSPACE_VIEW_COMPARE:
        if compare_summary.get("mode") == "model_compare" or selected_run_id == "__all__":
            return
    if current_view == WORKSPACE_VIEW_SINGLE_MODEL:
        if selected_run_id not in {"", "__all__"}:
            return

    if compare_summary.get("mode") == "model_compare" and selected_run_id == "__all__":
        set_workspace_view(WORKSPACE_VIEW_COMPARE)
    elif selected_run_id not in {"", "__all__"}:
        set_workspace_view(WORKSPACE_VIEW_SINGLE_MODEL)
    else:
        set_workspace_view(WORKSPACE_VIEW_HOME)


def clear_training_state() -> None:
    """Clear all transient frontend training state after completion."""
    st.session_state["ui_locked"] = False
    st.session_state["active_train_control"] = None
    st.session_state["training_experiment_id"] = None
    st.session_state["training_started_at_timestamp"] = None
    st.session_state["last_running_experiment_id"] = None
    st.session_state["current_auto_task_id"] = None
    st.session_state["auto_task_progress"] = None
    st.session_state["current_compare_task_id"] = None
    st.session_state["model_compare_task_progress"] = None
    st.session_state["skip_auto_poll_once"] = False
    st.session_state["manual_stop_requested"] = False


def clear_manual_suggestion_task_state() -> None:
    """Clear transient frontend state for one manual suggestion task."""
    st.session_state["manual_suggestion_task_id"] = None
    st.session_state["manual_suggestion_experiment_id"] = None
    st.session_state["manual_suggestion_status"] = None


def recover_active_auto_train_task_state() -> None:
    """Reattach frontend state to one backend auto-train task after a browser refresh."""
    if st.session_state.get("current_auto_task_id"):
        return
    ok, active_task_response = load_active_auto_train_task()
    if not ok:
        return
    st.session_state["current_auto_task_id"] = active_task_response["task_id"]
    st.session_state["active_train_control"] = "auto"
    st.session_state["ui_locked"] = active_task_response.get("status") in {"queued", "running", "stopping"}
    st.session_state["active_ai_panel_mode"] = "auto"
    existing_summary = st.session_state.get("auto_train_summary") or {}
    if existing_summary.get("mode") != "auto" or existing_summary.get("run_id") != active_task_response.get("run_id"):
        st.session_state["auto_train_summary"] = {
            "mode": "auto",
            "run_id": active_task_response.get("run_id"),
            "baseline": {},
            "rounds": [],
        }


def recover_active_model_compare_task_state() -> None:
    """Reattach frontend state to one backend model-compare task after a browser refresh."""
    if st.session_state.get("current_compare_task_id"):
        return
    ok, active_task_response = load_active_model_compare_task()
    if not ok:
        return
    st.session_state["current_compare_task_id"] = active_task_response["task_id"]
    st.session_state["active_train_control"] = "compare"
    st.session_state["ui_locked"] = active_task_response.get("status") in {"queued", "running"}
    st.session_state["selected_run_id"] = "__all__"
    if active_task_response.get("summary"):
        st.session_state["model_compare_summary"] = active_task_response["summary"]
    st.session_state["active_ai_panel_mode"] = "compare"
    st.session_state["compare_ai_panel"] = build_model_compare_ai_panel(
        active_task_response.get("summary"),
        task_status=active_task_response.get("status"),
        task_error=active_task_response.get("error"),
        progress={
            "elapsed_seconds": active_task_response.get("elapsed_seconds", 0.0),
            "current_model_name": active_task_response.get("current_model_name"),
            "current_model_index": active_task_response.get("current_model_index", 0),
            "total_models": active_task_response.get("total_models", 0),
        },
    )


def sync_model_compare_task_state() -> str | None:
    """Sync frontend state from one backend model-compare task snapshot."""
    current_compare_task_id = st.session_state.get("current_compare_task_id")
    if not current_compare_task_id:
        return None

    st.session_state["active_train_control"] = "compare"
    ok, compare_task_response = load_model_compare_task(current_compare_task_id)
    if not ok or "task_id" not in compare_task_response:
        return None

    st.session_state["activity_logs"] = compare_task_response.get("logs", [])
    if compare_task_response.get("summary") is not None:
        st.session_state["model_compare_summary"] = compare_task_response["summary"]
    st.session_state["selected_run_id"] = "__all__"
    st.session_state["training_experiment_id"] = compare_task_response.get("current_experiment_id")
    st.session_state["model_compare_task_progress"] = {
        "status": compare_task_response.get("status"),
        "elapsed_seconds": compare_task_response.get("elapsed_seconds", 0.0),
        "current_model_name": compare_task_response.get("current_model_name"),
        "current_model_index": compare_task_response.get("current_model_index", 0),
        "total_models": compare_task_response.get("total_models", 0),
        "current_run_id": compare_task_response.get("current_run_id"),
        "current_experiment_id": compare_task_response.get("current_experiment_id"),
    }
    st.session_state["active_ai_panel_mode"] = "compare"
    st.session_state["compare_ai_panel"] = build_model_compare_ai_panel(
        compare_task_response.get("summary"),
        task_status=compare_task_response.get("status"),
        task_error=compare_task_response.get("error"),
        progress=st.session_state["model_compare_task_progress"],
    )
    compare_status = compare_task_response.get("status")
    if compare_status in {"queued", "running"}:
        st.session_state["ui_locked"] = True
        return compare_status

    st.session_state["current_compare_task_id"] = None
    st.session_state["model_compare_task_progress"] = None
    st.session_state["training_experiment_id"] = None
    st.session_state["ui_locked"] = False
    st.session_state["active_train_control"] = None
    if compare_status == "success":
        set_post_action_notice("Model compare finished.")
    elif compare_status == "failed":
        set_post_action_notice(
            f"Model compare failed: {compare_task_response.get('error') or 'unknown error'}",
            "error",
        )
    return compare_status


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
        summary["task_error"] = auto_task_response.get("error")
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
        "elapsed_seconds": auto_task_response.get("elapsed_seconds", 0.0),
        "current_experiment_id": auto_task_response.get("current_experiment_id"),
        "latest_prompt_tokens_estimate": auto_task_response.get("latest_prompt_tokens_estimate"),
        "estimated_prompt_tokens_total": auto_task_response.get("estimated_prompt_tokens_total", 0),
        "latest_prompt_history_items": auto_task_response.get("latest_prompt_history_items"),
        "latest_provider_prompt_tokens": auto_task_response.get("latest_provider_prompt_tokens"),
        "latest_provider_completion_tokens": auto_task_response.get("latest_provider_completion_tokens"),
        "latest_provider_total_tokens": auto_task_response.get("latest_provider_total_tokens"),
        "provider_prompt_tokens_total": auto_task_response.get("provider_prompt_tokens_total", 0),
        "provider_completion_tokens_total": auto_task_response.get("provider_completion_tokens_total", 0),
        "provider_total_tokens_total": auto_task_response.get("provider_total_tokens_total", 0),
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


def sync_manual_training_state() -> None:
    """Sync one manually started experiment before rendering the main UI."""
    training_experiment_id = st.session_state.get("training_experiment_id")
    current_auto_task_id = st.session_state.get("current_auto_task_id")
    current_compare_task_id = st.session_state.get("current_compare_task_id")
    active_train_control = st.session_state.get("active_train_control")
    if not training_experiment_id or current_auto_task_id or current_compare_task_id or active_train_control in {"auto", "compare"}:
        return

    training_experiment_detail = load_experiment_detail(training_experiment_id)
    training_status = training_experiment_detail.get("status")
    if training_status == "running":
        if st.session_state.get("manual_stop_requested"):
            if st.session_state.get("last_stop_wait_experiment_id") != training_experiment_id:
                append_activity_log(
                    f"Stop requested for experiment {training_experiment_id}. Waiting for the worker to exit."
                )
                st.session_state["last_stop_wait_experiment_id"] = training_experiment_id
            return
        if st.session_state.get("last_running_experiment_id") != training_experiment_id:
            append_activity_log(f"Experiment {training_experiment_id} is running.")
            st.session_state["last_running_experiment_id"] = training_experiment_id
        return

    if st.session_state.get("last_finished_experiment_id") == training_experiment_id:
        return

    st.session_state["last_finished_experiment_id"] = training_experiment_id
    st.session_state["selected_experiment_id"] = training_experiment_id
    clear_training_state()
    if training_status == "discarded":
        clear_manual_suggestion_task_state()
        append_activity_log(f"Experiment {training_experiment_id} was discarded.")
        set_post_action_notice(f"Training stopped and discarded: {training_experiment_id}", "error")
        st.rerun()

    append_activity_log(
        f"Experiment {training_experiment_id} finished with status {training_status} "
        f"and {format_metric_summary(training_experiment_detail)}."
    )
    completed_run_id = training_experiment_detail.get("run_id")
    if training_status == "success" and completed_run_id:
        ok, suggestion_task_response = start_experiment_suggestion_task_request(training_experiment_id)
        if ok:
            st.session_state["manual_suggestion_task_id"] = suggestion_task_response["task_id"]
            st.session_state["manual_suggestion_experiment_id"] = training_experiment_id
            st.session_state["manual_suggestion_status"] = suggestion_task_response.get("status")
            st.session_state["manual_ai_panel"] = {
                "mode": "manual",
                "run_id": completed_run_id,
                "experiment_id": training_experiment_id,
                "result": build_result_snapshot(training_experiment_id, training_experiment_detail),
                "proposal": None,
                "suggestion_status": suggestion_task_response.get("status"),
            }
            st.session_state["active_ai_panel_mode"] = "manual"
            append_activity_log(
                f"Post-train AI suggestion requested for experiment {training_experiment_id}."
            )
        else:
            clear_manual_suggestion_task_state()
            append_activity_log(
                f'Post-train AI suggestion request failed: {suggestion_task_response.get("detail", suggestion_task_response)}'
            )
    else:
        clear_manual_suggestion_task_state()
    if training_status == "failed":
        if is_oom_failure(training_experiment_detail):
            set_post_action_notice(
                "Training failed: CUDA out of memory. "
                "Try a smaller batch size, or reduce image size if needed.",
                "error",
            )
        else:
            set_post_action_notice(f"Training failed with status: {training_status}", "error")
    else:
        set_post_action_notice(f"Training finished with status: {training_status}")
    st.rerun()


def sync_manual_suggestion_task_state() -> None:
    """Sync one background post-train suggestion task for manual mode."""
    experiment_id = st.session_state.get("manual_suggestion_experiment_id")
    if not experiment_id:
        return

    ok, suggestion_task = load_experiment_suggestion_task(experiment_id)
    if not ok:
        return

    previous_status = st.session_state.get("manual_suggestion_status")
    current_status = suggestion_task.get("status")
    st.session_state["manual_suggestion_task_id"] = suggestion_task.get("task_id")
    st.session_state["manual_suggestion_status"] = current_status

    manual_panel = dict(st.session_state.get("manual_ai_panel") or {})
    if manual_panel.get("mode") != "manual" or manual_panel.get("experiment_id") != experiment_id:
        experiment_detail = load_experiment_detail(experiment_id)
        manual_panel = {
            "mode": "manual",
            "run_id": experiment_detail.get("run_id"),
            "experiment_id": experiment_id,
            "result": build_result_snapshot(experiment_id, experiment_detail),
            "proposal": None,
        }
    manual_panel["suggestion_status"] = current_status
    manual_panel["suggestion_error"] = suggestion_task.get("error")
    manual_panel["token_usage"] = {
        "prompt_tokens": suggestion_task.get("latest_provider_prompt_tokens"),
        "completion_tokens": suggestion_task.get("latest_provider_completion_tokens"),
        "total_tokens": suggestion_task.get("latest_provider_total_tokens"),
    }

    if current_status == "success" and suggestion_task.get("suggestion"):
        experiment_detail = load_experiment_detail(experiment_id)
        manual_panel["result"] = build_result_snapshot(experiment_id, experiment_detail)
        manual_panel["proposal"] = suggestion_task["suggestion"]
        st.session_state["manual_ai_panel"] = manual_panel
        st.session_state["active_ai_panel_mode"] = "manual"
        if previous_status != "success":
            append_activity_log(f"Post-train AI suggestion is ready for experiment {experiment_id}.")
        clear_manual_suggestion_task_state()
        return

    st.session_state["manual_ai_panel"] = manual_panel
    st.session_state["active_ai_panel_mode"] = "manual"
    if current_status == "failed":
        if previous_status != "failed":
            append_activity_log(
                f'Post-train AI suggestion failed: {suggestion_task.get("error") or "unknown error"}'
            )
        clear_manual_suggestion_task_state()


@st.fragment(run_every="2s")
def render_live_training_monitor() -> None:
    """Keep the log and AI panel updated while training is active."""
    auto_task_id = st.session_state.get("current_auto_task_id")
    compare_task_id = st.session_state.get("current_compare_task_id")
    experiment_id = st.session_state.get("training_experiment_id")
    suggestion_experiment_id = st.session_state.get("manual_suggestion_experiment_id")
    if not auto_task_id and not compare_task_id and not experiment_id and not suggestion_experiment_id:
        return

    if auto_task_id:
        auto_status = sync_auto_train_task_state()
        refresh_activity_log_view()
        refresh_ai_panel_view()
        if st.session_state.pop("auto_result_refresh_needed", False):
            st.rerun()
        if auto_status in {"stopped", "stopped_by_policy", "failed"}:
            task_snapshot = request_json(
                f"/runs/auto-train/{auto_task_id}",
                {"error": "unknown error"},
            )
            if auto_status == "stopped":
                set_post_action_notice(
                    build_auto_train_terminal_message(
                        auto_status,
                        stop_reason=task_snapshot.get("stop_reason"),
                        task_error=task_snapshot.get("error"),
                    ),
                    "success",
                )
            elif auto_status == "stopped_by_policy":
                set_post_action_notice(
                    build_auto_train_terminal_message(
                        auto_status,
                        stop_reason=task_snapshot.get("stop_reason"),
                        task_error=task_snapshot.get("error"),
                    ),
                    "success",
                )
            elif auto_status == "failed":
                set_post_action_notice(
                    build_auto_train_terminal_message(
                        auto_status,
                        stop_reason=task_snapshot.get("stop_reason"),
                        task_error=task_snapshot.get("error"),
                    ),
                    "error",
                )
            clear_training_state()
            st.rerun()
        return

    if compare_task_id:
        compare_status = sync_model_compare_task_state()
        refresh_activity_log_view()
        refresh_ai_panel_view()
        if compare_status in {"success", "failed"}:
            st.rerun()
        return

    if suggestion_experiment_id and not experiment_id:
        sync_manual_suggestion_task_state()
        refresh_activity_log_view()
        refresh_ai_panel_view()
        return

    experiment_detail = load_experiment_detail(experiment_id)
    if experiment_detail.get("status") in {"success", "failed", "discarded"}:
        st.rerun()
        return

    if suggestion_experiment_id:
        sync_manual_suggestion_task_state()
        refresh_activity_log_view()
        refresh_ai_panel_view()


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
    selected_run_detail = load_run_detail(selected_run_id) if selected_run_id != "__all__" else None
    selected_run_model_name = (
        str(selected_run_detail.get("model_name"))
        if isinstance(selected_run_detail, dict) and selected_run_detail.get("model_name")
        else None
    )
    manual_train_action = resolve_manual_train_action(
        selected_run_id=selected_run_id,
        selected_run_model_name=selected_run_model_name,
        requested_model_name=payload["model_name"],
    )
    if manual_train_action == "create_run":
        if selected_run_id != "__all__" and selected_run_model_name != payload["model_name"]:
            append_activity_log(
                "Selected run model differs from the current panel. Creating a new run instead of appending."
            )
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
            resource = result.get("resource") or {}
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
                    "latency_ms": resource.get("latency_ms"),
                    "parameter_count_million": resource.get("parameter_count_million"),
                    "training_seconds": resource.get("training_seconds"),
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


def get_baseline_experiment_detail(run_id: str) -> dict[str, Any] | None:
    """Return the baseline experiment detail for one run."""
    run_summary = load_run_summary(run_id)
    baseline_experiment_id = run_summary.get("baseline_experiment_id")
    if baseline_experiment_id:
        return load_experiment_detail(baseline_experiment_id)

    run_detail = load_run_detail(run_id)
    experiments = run_detail.get("experiments", [])
    if not experiments:
        return None
    return load_experiment_detail(experiments[0]["id"])


def load_reference_config(selected_run_id: str) -> dict[str, Any]:
    """Load the latest config for the selected run or return defaults."""
    default_dataset = get_default_dataset_name()
    default_config = {
        "run_name": build_auto_run_name(default_dataset, "mobilenet_v3_small"),
        "dataset": default_dataset,
        "model_name": "mobilenet_v3_small",
        "use_demo_mode": True,
        "participates_in_ranking": True,
        "search_policy": default_search_policy(),
        "ranking_policy": default_ranking_policy(),
        "params": {
            "optimizer": "adamw",
            "learning_rate": 0.003,
            "batch_size": 64,
            "image_size": get_original_image_size(default_dataset, "mobilenet_v3_small"),
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
        "use_demo_mode": latest_experiment["config"].get("use_demo_mode", False),
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
            "use_demo_mode": bool(form_values.get("use_demo_mode", False)),
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
    st.session_state["training_started_at_timestamp"] = time.time()
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
    st.session_state["training_started_at_timestamp"] = time.time()
    return True, f'Appended and started experiment {experiment_response["id"]} under run {run_id}.'


def render_control_panel(
    runs: list[dict[str, Any]],
    selected_run_id: str,
    is_training_active: bool,
    *,
    workspace_mode: str,
) -> None:
    """Render the left-side parameter and action panel."""
    if workspace_mode == WORKSPACE_VIEW_COMPARE:
        st.subheader("Compare Setup")
        st.caption("这里定义 shared baseline 配置。当前 compare 会对内置候选模型全集执行同一套训练协议。")
    else:
        st.subheader("Optimize One Model")
        st.caption("这里围绕单个模型配置 baseline 和后续自动优化参数。")
    if is_training_active:
        st.warning("A training job is running. Actions are temporarily locked.")

    selected_run = next((run for run in runs if run["id"] == selected_run_id), None)
    has_existing_run = selected_run_id != "__all__" and selected_run is not None
    experiment_count = int(selected_run.get("experiment_count", 0)) if selected_run else 0
    selected_run_detail = load_run_detail(selected_run_id) if has_existing_run else None
    successful_experiment_count = (
        sum(1 for experiment in (selected_run_detail or {}).get("experiments", []) if experiment.get("status") == "success")
        if selected_run_detail is not None
        else 0
    )
    can_start_auto_train = has_existing_run and successful_experiment_count > 0

    reference_config = load_reference_config(selected_run_id)
    required_form_keys = {
        "run_name",
        "dataset",
        "model_name",
        "use_demo_mode",
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
        "allow_basic_hparam_search",
        "allow_strategy_search",
        "allow_loss_search",
        "allow_augmentation_search",
        "allow_model_module_search",
        "require_manual_approval_for_high_impact_changes",
        "use_default_ranking_policy",
        "ranking_primary_metric",
        "ranking_min_primary_metric_improvement",
        "ranking_primary_metric_parity_epsilon",
        "ranking_tie_breaker_metric",
        "ranking_min_tie_breaker_metric_improvement",
        "ranking_max_image_size",
        "based_on_experiment_ids",
    }
    should_reset_form_state = (
        st.session_state.get("form_reference_run") != selected_run_id
        or any(key not in st.session_state for key in required_form_keys)
    )
    if should_reset_form_state:
        st.session_state["run_name"] = reference_config["run_name"]
        st.session_state["dataset"] = reference_config["dataset"]
        st.session_state["model_name"] = reference_config["model_name"]
        st.session_state["use_demo_mode"] = bool(reference_config.get("use_demo_mode", False))
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
        st.session_state["allow_model_module_search"] = bool(search_policy.get("allow_model_module_search", False))
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

    with st.container(border=True):
        st.markdown("**Shared Config**" if workspace_mode == WORKSPACE_VIEW_COMPARE else "**Model Config**")
        if workspace_mode == WORKSPACE_VIEW_COMPARE:
            st.caption("Compare workspace 只保留跨模型共享的训练配置，不暴露单模型 run 相关字段。")
        else:
            st.caption("切换到已有 run 时，会自动带入该 run 最近一次实验的配置。")
        top_left, top_mid, top_right = st.columns(3)
        if workspace_mode == WORKSPACE_VIEW_COMPARE:
            with top_left:
                dataset_options = get_ready_dataset_options(st.session_state.get("dataset"))
                st.selectbox("Dataset", options=dataset_options, key="dataset")
            with top_mid:
                st.checkbox("Use Demo Subset", key="use_demo_mode")
            with top_right:
                st.caption("Compare All Models 会自动覆盖当前内置候选模型全集。")
            model_name = st.session_state["model_name"]
        else:
            with top_left:
                st.text_input("Run Name", key="run_name")
            with top_mid:
                dataset_options = get_ready_dataset_options(st.session_state.get("dataset"))
                st.selectbox("Dataset", options=dataset_options, key="dataset", on_change=sync_auto_run_name)
            with top_right:
                model_name = st.selectbox(
                    "Model",
                    options=list(SUPPORTED_MODELS.keys()),
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
            if workspace_mode == WORKSPACE_VIEW_COMPARE:
                st.caption("Compare 只跑 shared baseline，不自动进入第二阶段优化。")
            else:
                st.caption("Model Search 会先跑 baseline，再持续自动搜索，直到你手动停止或被策略停止。")
        with row_two[2]:
            if workspace_mode != WORKSPACE_VIEW_COMPARE:
                st.checkbox("Use Demo Subset", key="use_demo_mode")

        st.caption("Current Managed Params: " + summarize_ai_managed_params(st.session_state))
        if st.session_state.get("use_demo_mode"):
            st.caption("Mode: Demo subset training is enabled for this run.")
        else:
            st.caption("Mode: Full dataset training is enabled for this run.")

        if workspace_mode != WORKSPACE_VIEW_COMPARE:
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
                        st.session_state["allow_model_module_search"],
                    )
                )
                st.caption(
                    f"补充说明：image_size 保持原图大小时不搜索；当你手动改成非原图大小时，只在原图大小和当前设置之间搜索。"
                    f"当前原图大小：{original_image_size}；当前 image_size 搜索范围："
                    + ", ".join(str(choice) for choice in image_size_choices)
                )
                st.caption(
                    "Model module search 正在向 component-level 搜索收口。"
                    "当前 `MobileNetV3 Small` 已开放 `neck_name=avg_pool|gem_pool` 和 "
                    "`head_name=native_classifier|linear|dropout_linear`。"
                )
                st.checkbox(
                    "Allow basic hyperparameter search",
                    key="allow_basic_hparam_search",
                    disabled=is_training_active,
                )
                st.checkbox("Allow strategy search", key="allow_strategy_search", disabled=is_training_active)
                st.checkbox("Allow loss search", key="allow_loss_search", disabled=is_training_active)
                st.checkbox("Allow augmentation search", key="allow_augmentation_search", disabled=is_training_active)
                st.checkbox("Allow model module search", key="allow_model_module_search", disabled=is_training_active)
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
                            options=["latency_ms", "parameter_count_million", "val_loss", "top1_acc", "training_seconds"],
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
            "use_demo_mode": st.session_state["use_demo_mode"],
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
                "allow_model_module_search": st.session_state["allow_model_module_search"],
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

    model_search_button_label = (
        "Continue Optimizing Selected Model"
        if can_start_auto_train
        else "Optimize Selected Model"
    )
    active_train_control = st.session_state.get("active_train_control")
    auto_task_status = str((st.session_state.get("auto_task_progress") or {}).get("status") or "")
    auto_stop_requested = auto_task_status == "stopping"
    if workspace_mode == WORKSPACE_VIEW_COMPARE:
        compare_label = "Comparing..." if is_training_active and active_train_control == "compare" else "Compare All Models"
        compare_disabled = is_training_active
        if st.button(compare_label, disabled=compare_disabled, use_container_width=True):
            clear_manual_suggestion_task_state()
            st.session_state["active_train_control"] = "compare"
            st.session_state["active_ai_panel_mode"] = "compare"
            st.session_state["selected_run_id"] = "__all__"
            ok, response = start_model_compare_task_request(payload)
            if ok:
                st.session_state["current_compare_task_id"] = response["task_id"]
                st.session_state["model_compare_task_progress"] = {
                    "status": response.get("status"),
                    "elapsed_seconds": response.get("elapsed_seconds", 0.0),
                    "current_model_name": response.get("current_model_name"),
                    "current_model_index": response.get("current_model_index", 0),
                    "total_models": response.get("total_models", 0),
                    "current_run_id": response.get("current_run_id"),
                    "current_experiment_id": response.get("current_experiment_id"),
                }
                st.session_state["model_compare_summary"] = response.get("summary")
                st.session_state["compare_ai_panel"] = build_model_compare_ai_panel(
                    response.get("summary"),
                    task_status=response.get("status"),
                    task_error=response.get("error"),
                    progress=st.session_state["model_compare_task_progress"],
                )
                st.session_state["ui_locked"] = True
                append_activity_log(f"Model compare task started: {response['task_id']}")
                st.rerun()
            else:
                st.session_state["active_train_control"] = None
                st.error(f'Model compare failed to start: {response.get("detail", response)}')
        st.caption("当前 compare 会用统一 baseline 配置分别跑 `MobileNetV2 / MobileNetV3 Small / GoogLeNet`。")
        return

    action_label = (
        "Stopping..."
        if is_training_active and active_train_control == "auto" and auto_stop_requested
        else "Stop Training"
        if is_training_active and active_train_control == "auto"
        else model_search_button_label
    )
    action_disabled = auto_stop_requested or (is_training_active and active_train_control != "auto") or (
        not is_training_active and has_existing_run and not can_start_auto_train
    )
    if st.button(action_label, disabled=action_disabled, use_container_width=True):
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
            ok, response = start_auto_train_task_request(payload, selected_run_id)
            if ok:
                st.session_state["current_auto_task_id"] = response["task_id"]
                st.session_state["training_experiment_id"] = response.get("current_experiment_id")
                st.session_state["ui_locked"] = True
                st.session_state["skip_auto_poll_once"] = True
                st.session_state["active_ai_panel_mode"] = "auto"
                st.session_state["auto_train_summary"] = {
                    "mode": "auto",
                    "run_id": response.get("run_id"),
                    "baseline": {},
                    "rounds": [],
                    "task_status": response.get("status"),
                }
                append_activity_log(f"Model search task started: {response['task_id']}")
                st.rerun()
            else:
                st.session_state["active_train_control"] = None
                st.error(f'Model search failed to start: {response.get("detail", response)}')
    if not has_existing_run:
        st.caption("当前会基于表单配置创建一个新的单模型 run，并自动从 baseline 进入后续优化。")
    elif not can_start_auto_train:
        st.caption("当前 run 还没有成功 baseline。`Optimize Selected Model` 会自动补 baseline 并继续搜索。")
    else:
        st.caption("当前会基于所选 run 的已有实验继续优化，不会重新创建 baseline。")


def render_activity_log() -> None:
    """Render the training activity log."""
    global LIVE_LOG_CONTAINER
    st.subheader("Execution Log")
    st.caption("这里保留执行、重试、停止和异常事件；每轮策略细节放到右侧查看。")
    LIVE_LOG_CONTAINER = st.empty()
    refresh_activity_log_view()


def render_ai_suggestion_panel() -> None:
    """Render the latest AI suggestion card."""
    global LIVE_AI_PANEL_CONTAINER
    st.subheader("AI Suggestions")
    st.caption("这里展示 Model Search 总结，以及 Models Compare 结束后的推荐动作。")
    LIVE_AI_PANEL_CONTAINER = st.empty()
    refresh_ai_panel_view()


def render_workspace_header(title: str, description: str, *, show_back_button: bool = True) -> None:
    """Render one workspace header row."""
    header_left, header_right = st.columns([8, 1.5])
    with header_left:
        st.subheader(title)
        st.caption(description)
    with header_right:
        st.caption(" ")
        if show_back_button and st.button("Back Home", key=f"back_home_{title}", use_container_width=True):
            set_workspace_view(WORKSPACE_VIEW_HOME)
            st.rerun()


def render_home_view(runs: list[dict[str, Any]], is_training_active: bool) -> None:
    """Render the lightweight workspace entry page."""
    st.markdown("## Choose Your Workflow")
    st.caption("先决定你当前要做的是单模型优化，还是全模型横向比较。进入工作台后再看详细配置、运行状态和结果。")

    active_task_parts: list[str] = []
    auto_task_progress = st.session_state.get("auto_task_progress") or {}
    if st.session_state.get("current_auto_task_id"):
        active_task_parts.append(
            "单模型优化进行中"
            if str(auto_task_progress.get("status") or "") in {"queued", "running", "stopping"}
            else "存在最近的单模型优化记录"
        )
    compare_task_progress = st.session_state.get("model_compare_task_progress") or {}
    if st.session_state.get("current_compare_task_id"):
        active_task_parts.append(
            "全模型比较进行中"
            if str(compare_task_progress.get("status") or "") in {"queued", "running"}
            else "存在最近的全模型比较记录"
        )
    if active_task_parts:
        st.info(" / ".join(active_task_parts))
    elif is_training_active:
        st.info("当前存在运行中的任务。")

    runs_with_history = [run for run in runs if int(run.get("experiment_count", 0)) > 0]
    compare_summary = st.session_state.get("model_compare_summary") or {}
    latest_compare_count = len(compare_summary.get("candidate_results") or []) if compare_summary.get("mode") == "model_compare" else 0

    card_left, card_right = st.columns(2, gap="large")
    with card_left:
        with st.container(border=True):
            st.markdown("### Optimize One Model")
            st.caption("已经决定模型时，从这里进入单模型优化工作台。系统会自动创建或复用 run，并从 baseline 进入后续搜索。")
            metric_columns = st.columns(2)
            metric_columns[0].metric("Runs With History", len(runs_with_history))
            metric_columns[1].metric("Selected Run", "Ready" if str(st.session_state.get("selected_run_id") or "__all__") not in {"", "__all__"} else "New")
            if st.button("Open Single-Model Workspace", key="open_single_workspace", use_container_width=True):
                set_workspace_view(WORKSPACE_VIEW_SINGLE_MODEL)
                st.rerun()
    with card_right:
        with st.container(border=True):
            st.markdown("### Compare All Models")
            st.caption("还没决定模型时，从这里进入 compare 工作台。系统会对当前内置候选模型全集执行 shared baseline 比较。")
            metric_columns = st.columns(2)
            metric_columns[0].metric("Built-in Candidates", 3)
            metric_columns[1].metric("Latest Compare", latest_compare_count if latest_compare_count else "None")
            if st.button("Open Compare Workspace", key="open_compare_workspace", use_container_width=True):
                st.session_state["selected_run_id"] = "__all__"
                set_workspace_view(WORKSPACE_VIEW_COMPARE)
                st.rerun()


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
    st.caption("这里用于横向比较实验结果和参数；上面的 Round History 用于解释每轮为什么这样改。")
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
            "latency_ms",
            "parameter_count_million",
            "training_seconds",
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


def render_model_compare_results() -> None:
    """Render one cross-model compare summary when available."""
    compare_summary = st.session_state.get("model_compare_summary") or {}
    if compare_summary.get("mode") != "model_compare":
        st.caption("Run `Compare All Models` to inspect cross-model shared-baseline results.")
        return

    candidate_results = list(compare_summary.get("candidate_results") or [])
    shared_baseline_config = compare_summary.get("shared_baseline_config") or {}
    progress = st.session_state.get("model_compare_task_progress") or {}
    compare_status = progress.get("status") or "success"

    st.subheader("Models Compare")
    if compare_status in {"queued", "running"}:
        current_model_name = progress.get("current_model_name") or "-"
        current_model_index = int(progress.get("current_model_index") or 0)
        total_models = int(progress.get("total_models") or 0)
        elapsed_seconds = float(progress.get("elapsed_seconds") or 0.0)
        st.caption(
            f"Comparing {current_model_name} ({current_model_index}/{total_models}) | "
            f"elapsed {format_elapsed_seconds(elapsed_seconds)}"
        )
    else:
        st.caption("Cross-model shared-baseline comparison result.")

    with st.expander("Shared Baseline Config", expanded=False):
        st.json(shared_baseline_config)

    result_rows = [
        {
            "model_name": result.get("model_name"),
            "status": result.get("status"),
            "top1_acc": result.get("top1_acc"),
            "latency_ms": result.get("latency_ms"),
            "parameter_count_million": result.get("parameter_count_million"),
            "run_id": result.get("run_id"),
            "baseline_experiment_id": result.get("baseline_experiment_id"),
            "normalized_config_notes": "; ".join(result.get("normalized_config_notes") or []),
        }
        for result in candidate_results
    ]
    if not result_rows:
        st.info("No model compare results yet.")
        return

    compare_df = pd.DataFrame(result_rows)
    chart_df = compare_df.dropna(subset=["latency_ms", "top1_acc"]).copy()
    if not chart_df.empty:
        scatter_chart = (
            alt.Chart(chart_df)
            .mark_circle(size=160)
            .encode(
                x=alt.X("latency_ms:Q", title="Latency (ms)"),
                y=alt.Y("top1_acc:Q", title="Top1 Acc"),
                color=alt.Color("model_name:N", legend=alt.Legend(title="Model")),
                tooltip=[
                    alt.Tooltip("model_name:N", title="Model"),
                    alt.Tooltip("top1_acc:Q", title="Top1 Acc", format=".4f"),
                    alt.Tooltip("latency_ms:Q", title="Latency (ms)", format=".3f"),
                    alt.Tooltip("parameter_count_million:Q", title="Params (M)", format=".3f"),
                    alt.Tooltip("status:N", title="Status"),
                ],
            )
            .properties(height=320)
        )
        label_chart = scatter_chart.mark_text(align="left", dx=8, dy=-8).encode(text="model_name:N")
        st.altair_chart(scatter_chart + label_chart, use_container_width=True)
    else:
        st.caption("Scatter plot will appear after completed candidates report both latency and accuracy.")

    st.dataframe(
        compare_df[
            [
                "model_name",
                "status",
                "top1_acc",
                "latency_ms",
                "parameter_count_million",
                "run_id",
                "normalized_config_notes",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    selectable_candidates = [
        result for result in candidate_results
        if result.get("status") == "success" and result.get("run_id")
    ]
    if compare_status in {"queued", "running"} or not selectable_candidates:
        return

    selected_compare_run_id = st.selectbox(
        "Choose One Model To Optimize",
        options=[str(result["run_id"]) for result in selectable_candidates],
        format_func=lambda run_id: next(
            (
                f"{result['model_name']} | acc={format_metric_value('top1_acc', result.get('top1_acc'))} | "
                f"latency={format_metric_value('latency_ms', result.get('latency_ms'))} ms"
                for result in selectable_candidates
                if result.get("run_id") == run_id
            ),
            run_id,
        ),
        key="selected_model_compare_run_id",
    )
    if st.button("Optimize This Model", use_container_width=True):
        selected_result = next(
            result for result in selectable_candidates if result.get("run_id") == selected_compare_run_id
        )
        st.session_state["selected_run_id"] = selected_compare_run_id
        st.session_state["selected_experiment_id"] = selected_result.get("baseline_experiment_id")
        set_post_action_notice(f"Selected {selected_result['model_name']} for single-model optimization.")
        st.rerun()


def render_result_summary(selected_run_id: str) -> None:
    """Render the result summary cards for one selected run."""
    def render_result_card(
        title: str,
        experiment_detail: dict[str, Any] | None,
        *,
        emphasize_best: bool = False,
    ) -> None:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            if experiment_detail is None:
                st.caption("No experiment yet.")
                return

            experiment_id = experiment_detail.get("id")
            status = experiment_detail.get("status", "-")
            metrics = (experiment_detail.get("result") or {}).get("metrics") or {}
            resource = (experiment_detail.get("result") or {}).get("resource") or {}
            elapsed_seconds_override = experiment_detail.get("_elapsed_seconds_override")
            running_elapsed_seconds = get_running_elapsed_seconds(
                str(experiment_detail.get("run_id") or ""),
                str(experiment_id or ""),
            )
            if isinstance(elapsed_seconds_override, (int, float)):
                running_elapsed_seconds = float(elapsed_seconds_override)
            timing_label = (
                format_elapsed_seconds(running_elapsed_seconds)
                if status in {"running", "queued", "stopping"} and running_elapsed_seconds is not None
                else get_experiment_training_time_label(experiment_detail)
            )
            time_label = "Running Time" if status in {"running", "queued", "stopping"} else "Training Time"

            summary_row = st.columns(4)
            summary_row[0].metric("Experiment", get_short_experiment_id(experiment_id))
            summary_row[1].metric("Status", status)
            summary_row[2].metric(time_label, timing_label)
            summary_row[3].metric("Best Epoch", format_metric_value("best_epoch", metrics.get("best_epoch")))

            comparison_row = st.columns(4)
            comparison_row[0].metric("Top1 Acc", format_metric_value("top1_acc", metrics.get("top1_acc")))
            comparison_row[1].metric("Val Loss", format_metric_value("val_loss", metrics.get("val_loss")))
            comparison_row[2].metric("Latency (ms)", format_metric_value("latency_ms", resource.get("latency_ms")))
            comparison_row[3].metric("Params (M)", format_metric_value("parameter_count_million", resource.get("parameter_count_million")))

            if emphasize_best:
                st.caption("This card tracks the current best experiment in the run.")

    if selected_run_id in {"", "__all__"}:
        st.caption("Select or create a run to inspect results.")
    else:
        baseline_experiment_detail = get_baseline_experiment_detail(selected_run_id)
        best_experiment_detail = get_best_experiment_detail(selected_run_id)

        is_baseline_running = (
            baseline_experiment_detail is not None
            and baseline_experiment_detail.get("status") == "running"
            and st.session_state.get("training_experiment_id") == baseline_experiment_detail.get("id")
        )
        should_show_best_card = (
            best_experiment_detail is not None
            and baseline_experiment_detail is not None
            and best_experiment_detail.get("id") != baseline_experiment_detail.get("id")
        )

        render_result_card("Baseline", baseline_experiment_detail)
        if not is_baseline_running and should_show_best_card:
            render_result_card("Best", best_experiment_detail, emphasize_best=True)


def should_refresh_result_summary(selected_run_id: str) -> bool:
    """Return whether the summary cards should refresh automatically."""
    if selected_run_id in {"", "__all__"}:
        return False
    auto_task_progress = st.session_state.get("auto_task_progress") or {}
    auto_task_status = str(auto_task_progress.get("status") or "")
    if (
        st.session_state.get("current_auto_task_id")
        and st.session_state.get("selected_run_id") == selected_run_id
        and auto_task_status in {"queued", "running", "stopping"}
    ):
        return True
    training_experiment_id = st.session_state.get("training_experiment_id")
    if not training_experiment_id:
        return False
    training_experiment_detail = load_experiment_detail(training_experiment_id)
    return (
        training_experiment_detail.get("run_id") == selected_run_id
        and training_experiment_detail.get("status") == "running"
    )


@st.fragment(run_every="2s")
def render_live_result_summary(selected_run_id: str) -> None:
    """Refresh the result summary cards while training is active."""
    render_result_summary(selected_run_id)


def render_result_workspace(runs: list[dict[str, Any]], selected_run_id: str) -> None:
    """Render the right-side result workspace."""
    compare_summary = st.session_state.get("model_compare_summary") or {}
    if selected_run_id == "__all__" and compare_summary.get("mode") == "model_compare":
        render_model_compare_results()
        return

    st.subheader("Results")
    if should_refresh_result_summary(selected_run_id):
        render_live_result_summary(selected_run_id)
    else:
        render_result_summary(selected_run_id)

    render_ai_suggestion_panel()
    auto_task_progress = st.session_state.get("auto_task_progress") or {}
    auto_task_status = str(auto_task_progress.get("status") or "")
    current_auto_run_id = st.session_state.get("selected_run_id")
    is_selected_run_auto_training = (
        selected_run_id not in {"", "__all__"}
        and current_auto_run_id == selected_run_id
        and st.session_state.get("current_auto_task_id")
        and auto_task_status in {"queued", "running", "stopping"}
    )
    selected_run = next((run for run in runs if run["id"] == selected_run_id), None)
    selected_run_experiment_count = int(selected_run.get("experiment_count", 0)) if selected_run else 0
    has_auto_train_history = selected_run_experiment_count > 1
    if is_selected_run_auto_training:
        st.subheader("Training Records")
        st.caption("Model Search 进行中时先隐藏训练记录表，结束后再统一查看和比较。")
    elif selected_run_id not in {"", "__all__"} and not has_auto_train_history:
        st.subheader("Training Records")
        st.caption("当前只有 baseline，暂不显示训练记录表；进入 Model Search 后再展示历史对比。")
    else:
        render_training_records_workspace(runs, selected_run_id)


def main() -> None:
    """Render the Streamlit demo UI."""
    global REQUEST_CACHE
    REQUEST_CACHE = {}
    st.set_page_config(page_title="AutoVisionLab Demo", layout="wide")
    st.title("AutoVisionLab Demo")
    st.caption("左侧做单模型优化或全模型比较，右侧查看结果、总结和下一步建议。")
    st.session_state["dataset_catalog"] = load_dataset_catalog()
    post_action_notice = st.session_state.pop("post_action_notice", None)
    if post_action_notice:
        if post_action_notice["level"] == "success":
            st.success(post_action_notice["message"])
        else:
            st.error(post_action_notice["message"])

    recover_active_model_compare_task_state()
    recover_active_auto_train_task_state()
    sync_model_compare_task_state()
    sync_auto_train_task_state()
    sync_manual_training_state()

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


if __name__ == "__main__":
    main()
