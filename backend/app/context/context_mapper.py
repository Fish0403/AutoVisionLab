"""Map run and experiment payloads into prompt-context schemas."""

from __future__ import annotations

from typing import Any

from app.schemas.prompt_context import (
    BasePromptPayload,
    ConfigDeltaItem,
    HistoryItemSummary,
    ResourceSnapshot,
    ResultSnapshot,
    RetryPromptPayload,
    MetricSnapshot,
    SourcePromptPayload,
)


ALLOWED_CHANGE_FIELDS = (
    "optimizer",
    "learning_rate",
    "batch_size",
    "image_size",
    "epochs",
    "weight_decay",
    "scheduler",
    "label_smoothing",
    "mixup_alpha",
    "cutmix_alpha",
    "random_erasing_prob",
    "loss_name",
    "focal_gamma",
    "aux_logits",
    "neck_name",
    "head_name",
)


def _extract_params(experiment_payload: dict[str, Any]) -> dict[str, Any]:
    config_payload = experiment_payload.get("config") or experiment_payload.get("experiment_config") or {}
    params_payload = config_payload.get("params")
    if isinstance(params_payload, dict):
        return params_payload
    top_level_params = experiment_payload.get("params")
    if isinstance(top_level_params, dict):
        return top_level_params
    return {}


def _extract_train_hyp(experiment_payload: dict[str, Any]) -> dict[str, Any]:
    config_payload = experiment_payload.get("config") or experiment_payload.get("experiment_config") or {}
    train_hyp_payload = config_payload.get("train_hyp")
    if isinstance(train_hyp_payload, dict):
        return train_hyp_payload
    top_level_train_hyp = experiment_payload.get("train_hyp")
    if isinstance(top_level_train_hyp, dict):
        return top_level_train_hyp
    return {}


def _extract_model_recipe(experiment_payload: dict[str, Any]) -> dict[str, Any]:
    config_payload = experiment_payload.get("config") or experiment_payload.get("experiment_config") or {}
    model_recipe_payload = config_payload.get("model_recipe")
    if isinstance(model_recipe_payload, dict):
        return model_recipe_payload
    top_level_model_recipe = experiment_payload.get("model_recipe")
    if isinstance(top_level_model_recipe, dict):
        return top_level_model_recipe
    return {}


def _extract_result(experiment_payload: dict[str, Any]) -> dict[str, Any]:
    result_payload = experiment_payload.get("result")
    if isinstance(result_payload, dict):
        return result_payload
    metrics_payload = experiment_payload.get("metrics")
    resource_payload = experiment_payload.get("resource")
    if isinstance(metrics_payload, dict) or isinstance(resource_payload, dict):
        return {
            "status": experiment_payload.get("status"),
            "metrics": metrics_payload if isinstance(metrics_payload, dict) else {},
            "resource": resource_payload if isinstance(resource_payload, dict) else {},
        }
    return {}


def _extract_summary(experiment_payload: dict[str, Any]) -> str | None:
    summary = experiment_payload.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    decision_reason = experiment_payload.get("decision_reason")
    if isinstance(decision_reason, str) and decision_reason.strip():
        return decision_reason.strip()
    return None


def _normalize_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    return None


def _build_change_type(from_value: Any, to_value: Any) -> str:
    if isinstance(from_value, bool) and isinstance(to_value, bool):
        return "toggle"
    if isinstance(from_value, (int, float)) and isinstance(to_value, (int, float)):
        return "increase" if to_value > from_value else "decrease"
    return "switch"


def build_delta_items(
    previous_params: dict[str, Any],
    current_params: dict[str, Any],
) -> list[ConfigDeltaItem]:
    """Build structured delta items between two parameter snapshots."""
    delta_items: list[ConfigDeltaItem] = []
    field_names = sorted(set(previous_params) | set(current_params))
    for field_name in field_names:
        previous_value = previous_params.get(field_name)
        current_value = current_params.get(field_name)
        if previous_value == current_value:
            continue
        delta_items.append(
            ConfigDeltaItem(
                field=field_name,
                from_value=previous_value,
                to_value=current_value,
                change_type=_build_change_type(previous_value, current_value),
            )
        )
    return delta_items


def build_result_snapshot(experiment_payload: dict[str, Any]) -> ResultSnapshot:
    """Build one compact result snapshot from an experiment payload."""
    result_payload = _extract_result(experiment_payload)
    metrics_payload = result_payload.get("metrics") if isinstance(result_payload.get("metrics"), dict) else {}
    resource_payload = result_payload.get("resource") if isinstance(result_payload.get("resource"), dict) else {}
    return ResultSnapshot(
        status=result_payload.get("status") or experiment_payload.get("status"),
        metrics=MetricSnapshot(
            top1_acc=_normalize_number(metrics_payload.get("top1_acc")),
            val_loss=_normalize_number(metrics_payload.get("val_loss")),
            train_loss=_normalize_number(metrics_payload.get("train_loss")),
            best_epoch=metrics_payload.get("best_epoch"),
            latency_ms=_normalize_number(resource_payload.get("latency_ms")),
            parameter_count_million=_normalize_number(resource_payload.get("parameter_count_million")),
        ),
        resource=ResourceSnapshot(
            training_seconds=resource_payload.get("training_seconds"),
            gpu_memory_mb=resource_payload.get("gpu_memory_mb"),
        ),
    )


def build_base_prompt_payload(experiment_payload: dict[str, Any]) -> BasePromptPayload:
    """Build the baseline prompt payload."""
    return BasePromptPayload(
        experiment_id=str(experiment_payload.get("id")),
        params=_extract_params(experiment_payload),
        train_hyp=_extract_train_hyp(experiment_payload),
        model_recipe=_extract_model_recipe(experiment_payload),
        result_snapshot=build_result_snapshot(experiment_payload),
        summary=_extract_summary(experiment_payload),
    )


def build_source_prompt_payload(
    *,
    base_experiment_payload: dict[str, Any],
    source_experiment_payload: dict[str, Any],
    is_best: bool,
) -> SourcePromptPayload:
    """Build the source prompt payload relative to the baseline."""
    return SourcePromptPayload(
        experiment_id=str(source_experiment_payload.get("id")),
        based_on_experiment_id=str(base_experiment_payload.get("id")),
        delta_from_base=build_delta_items(
            _extract_params(base_experiment_payload),
            _extract_params(source_experiment_payload),
        ),
        result_snapshot=build_result_snapshot(source_experiment_payload),
        is_best=is_best,
        summary=_extract_summary(source_experiment_payload),
    )


def build_history_item_summary(
    *,
    source_experiment_payload: dict[str, Any],
    experiment_payload: dict[str, Any],
) -> HistoryItemSummary:
    """Build one source-relative attempt summary."""
    error_summary = experiment_payload.get("error_summary")
    if not isinstance(error_summary, str) or not error_summary.strip():
        error_summary = experiment_payload.get("error")
    if not isinstance(error_summary, str) or not error_summary.strip():
        error_summary = None
    return HistoryItemSummary(
        experiment_id=str(experiment_payload.get("id")),
        based_on_source_experiment_id=str(source_experiment_payload.get("id")),
        delta_from_source=build_delta_items(
            _extract_params(source_experiment_payload),
            _extract_params(experiment_payload),
        ),
        decision=experiment_payload.get("decision"),
        result_snapshot=build_result_snapshot(experiment_payload),
        outcome_summary=_extract_summary(experiment_payload),
        execution_error_summary=error_summary,
    )


def build_retry_prompt_payload(retry_feedback: str) -> RetryPromptPayload:
    """Build one retry feedback payload for a rejected proposal."""
    invalid_fields: list[str] = []
    for field_name in ALLOWED_CHANGE_FIELDS:
        if field_name in retry_feedback:
            invalid_fields.append(field_name)
    return RetryPromptPayload(
        proposal_rejection_reason=retry_feedback,
        invalid_fields=sorted(set(invalid_fields)),
        retry_guidance="请选择一个合法且可执行的下一步，并避免重复被拒绝的方向。",
    )
