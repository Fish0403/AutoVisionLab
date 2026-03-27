"""Helpers for centralized run policy decisions."""

from __future__ import annotations

from typing import Any

from app.models.experiment import ExperimentModel
from app.schemas.run_policy import RunPolicy
from app.services.parameter_space import (
    AUGMENTATION_SEARCH_FIELDS,
    BASIC_HPARAM_SEARCH_FIELDS,
    LOSS_SEARCH_FIELDS,
    STRATEGY_SEARCH_FIELDS,
)


ALL_SEARCH_DIMENSIONS = {"basic", "augmentation", "loss", "strategy"}


def get_default_run_policy() -> RunPolicy:
    """Return the default run policy used across the backend."""
    return RunPolicy()


def count_consecutive_stagnation_rounds(experiment_history: list[dict[str, Any]]) -> int:
    """Count consecutive terminal experiments since the latest promoted keep result."""
    stagnation_rounds = 0
    for experiment in reversed(experiment_history):
        if experiment.get("decision") == "keep":
            break
        if experiment.get("status") in {"success", "failed", "discarded"}:
            stagnation_rounds += 1
    return stagnation_rounds


def _get_stagnation_history(experiment_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return trailing terminal experiments since the latest promoted keep result."""
    stagnation_history: list[dict[str, Any]] = []
    for experiment in reversed(experiment_history):
        if experiment.get("decision") == "keep":
            break
        if experiment.get("status") in {"success", "failed", "discarded"}:
            stagnation_history.append(experiment)
    stagnation_history.reverse()
    return stagnation_history


def _get_effective_history_change_fields(experiment: dict[str, Any]) -> set[str]:
    """Return concrete changed fields recorded in one experiment proposal."""
    proposal_payload = experiment.get("proposal") or {}
    changes = proposal_payload.get("changes") or {}
    return {
        field_name
        for field_name, value in changes.items()
        if value is not None
    }


def get_field_dimension(field_name: str) -> str | None:
    """Return the search dimension for one field."""
    if field_name in BASIC_HPARAM_SEARCH_FIELDS:
        return "basic"
    if field_name in AUGMENTATION_SEARCH_FIELDS:
        return "augmentation"
    if field_name in LOSS_SEARCH_FIELDS:
        return "loss"
    if field_name in STRATEGY_SEARCH_FIELDS:
        return "strategy"
    return None


def get_change_dimensions(changed_fields: set[str]) -> set[str]:
    """Return the set of search dimensions touched by the given fields."""
    return {
        dimension
        for dimension in (get_field_dimension(field_name) for field_name in changed_fields)
        if dimension is not None
    }


def determine_change_budget(
    experiment_history: list[dict[str, Any]],
    *,
    policy: RunPolicy | None = None,
) -> tuple[int, int]:
    """Return consecutive stagnation rounds and the current proposal field budget."""
    effective_policy = policy or get_default_run_policy()
    stagnation_rounds = count_consecutive_stagnation_rounds(experiment_history)
    max_changed_fields = effective_policy.default_max_changed_fields
    if stagnation_rounds >= effective_policy.stagnation_rounds_for_combined_changes:
        max_changed_fields = effective_policy.max_changed_fields_after_stagnation
    return stagnation_rounds, max_changed_fields


def determine_temporarily_blocked_fields(
    experiment_history: list[dict[str, Any]],
    *,
    policy: RunPolicy | None = None,
) -> set[str]:
    """Return fields that should cool down after repeated failed use."""
    effective_policy = policy or get_default_run_policy()
    stagnation_history = _get_stagnation_history(experiment_history)
    latest_index = len(stagnation_history) - 1
    if latest_index < 0:
        return set()

    blocked_fields: set[str] = set()
    all_fields = {
        field_name
        for experiment in stagnation_history
        for field_name in _get_effective_history_change_fields(experiment)
    }
    for field_name in all_fields:
        streak_length = 0
        cooldown_anchor_index: int | None = None
        for index, experiment in enumerate(stagnation_history):
            changed_fields = _get_effective_history_change_fields(experiment)
            if field_name in changed_fields:
                streak_length += 1
                if streak_length >= effective_policy.consecutive_failures_before_field_cooldown:
                    cooldown_anchor_index = index
            else:
                streak_length = 0
        if cooldown_anchor_index is None:
            continue
        if latest_index - cooldown_anchor_index < effective_policy.field_cooldown_rounds:
            blocked_fields.add(field_name)
    return blocked_fields


def determine_forbidden_dimensions(
    experiment_history: list[dict[str, Any]],
    *,
    policy: RunPolicy | None = None,
) -> set[str]:
    """Return dimensions that the next proposal should switch away from."""
    effective_policy = policy or get_default_run_policy()
    stagnation_history = _get_stagnation_history(experiment_history)
    if len(stagnation_history) < effective_policy.stagnation_rounds_for_dimension_switch:
        return set()

    recent_history = stagnation_history[-effective_policy.stagnation_rounds_for_dimension_switch :]
    recent_dimension_sets = [
        get_change_dimensions(_get_effective_history_change_fields(experiment))
        for experiment in recent_history
    ]
    if not recent_dimension_sets or any(not dimensions for dimensions in recent_dimension_sets):
        return set()

    shared_dimensions = set.intersection(*recent_dimension_sets)
    if len(shared_dimensions) == 1:
        return shared_dimensions
    return set()


def proposal_switches_dimension(
    changed_fields: set[str],
    *,
    forbidden_dimensions: set[str],
) -> bool:
    """Return whether the proposal escapes the currently forbidden dimensions."""
    if not forbidden_dimensions:
        return True
    proposal_dimensions = get_change_dimensions(changed_fields)
    if not proposal_dimensions:
        return False
    return not proposal_dimensions.issubset(forbidden_dimensions)


def require_non_basic_change_for_round(
    round_index: int,
    total_rounds: int,
    *,
    policy: RunPolicy | None = None,
) -> bool:
    """Return whether the current round must include augmentation/loss/strategy changes."""
    if total_rounds <= 1:
        return False
    effective_policy = policy or get_default_run_policy()
    threshold_round = total_rounds * effective_policy.auto_train_non_basic_change_after_round_ratio
    return round_index > threshold_round


def extract_ranking_metrics(experiment: ExperimentModel) -> tuple[float | None, float | None]:
    """Return the key scalar metrics used by the promotion policy."""
    result = experiment.result or {}
    metrics = result.get("metrics") or {}
    top1_acc = metrics.get("top1_acc")
    val_loss = metrics.get("val_loss")
    return (
        float(top1_acc) if isinstance(top1_acc, (int, float)) else None,
        float(val_loss) if isinstance(val_loss, (int, float)) else None,
    )


def format_metric_value(value: float | None) -> str:
    """Format one scalar metric for decision logs."""
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def evaluate_promotion(
    candidate: ExperimentModel,
    incumbent: ExperimentModel | None,
    *,
    policy: RunPolicy | None = None,
) -> tuple[bool, str]:
    """Return whether the candidate should be promoted over the incumbent."""
    effective_policy = policy or get_default_run_policy()
    candidate_top1_acc, candidate_val_loss = extract_ranking_metrics(candidate)
    if incumbent is None:
        return True, "Promoted as the first successful experiment in the run."

    incumbent_top1_acc, incumbent_val_loss = extract_ranking_metrics(incumbent)
    if incumbent_top1_acc is not None and candidate_top1_acc is not None:
        top1_delta = candidate_top1_acc - incumbent_top1_acc
        if top1_delta >= effective_policy.min_top1_acc_promotion_delta:
            return (
                True,
                "Promoted by run policy: "
                f"top1_acc improved from {format_metric_value(incumbent_top1_acc)} "
                f"to {format_metric_value(candidate_top1_acc)}.",
            )
        if abs(top1_delta) <= effective_policy.top1_acc_parity_epsilon:
            if incumbent_val_loss is not None and candidate_val_loss is not None:
                val_loss_delta = incumbent_val_loss - candidate_val_loss
                if val_loss_delta >= effective_policy.min_val_loss_promotion_delta:
                    return (
                        True,
                        "Promoted by run policy: "
                        f"val_loss improved from {format_metric_value(incumbent_val_loss)} "
                        f"to {format_metric_value(candidate_val_loss)} at comparable top1_acc.",
                    )
                return (
                    False,
                    "Discarded by run policy: comparable top1_acc but val_loss "
                    f"did not improve by at least {effective_policy.min_val_loss_promotion_delta:.3f}.",
                )
        return (
            False,
            "Discarded by run policy: top1_acc "
            f"({format_metric_value(candidate_top1_acc)}) did not beat current best "
            f"({format_metric_value(incumbent_top1_acc)}) by at least "
            f"{effective_policy.min_top1_acc_promotion_delta:.3f}.",
        )

    if incumbent_top1_acc is None and candidate_top1_acc is not None:
        return True, "Promoted by run policy: candidate reports top1_acc while the incumbent does not."

    if incumbent_top1_acc is None and candidate_top1_acc is None:
        if incumbent_val_loss is not None and candidate_val_loss is not None:
            val_loss_delta = incumbent_val_loss - candidate_val_loss
            if val_loss_delta >= effective_policy.min_val_loss_promotion_delta:
                return (
                    True,
                    "Promoted by run policy: "
                    f"val_loss improved from {format_metric_value(incumbent_val_loss)} "
                    f"to {format_metric_value(candidate_val_loss)}.",
                )
        return (
            False,
            "Discarded by run policy: no significant improvement was observed in ranking metrics.",
        )

    return (
        False,
        "Discarded by run policy: candidate does not provide enough ranking evidence to beat the current best.",
    )
