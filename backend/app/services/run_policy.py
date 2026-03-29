"""Helpers for centralized run policy decisions."""

from __future__ import annotations

from typing import Any

from app.models.experiment import ExperimentModel
from app.schemas.parameter_space import SearchPolicy
from app.schemas.ranking_policy import RankingMetricMode, RankingPolicy
from app.schemas.run_policy import RunPolicy
from app.services.parameter_space import (
    AUGMENTATION_SEARCH_FIELDS,
    BASIC_HPARAM_SEARCH_FIELDS,
    LOSS_SEARCH_FIELDS,
    MODEL_MODULE_SEARCH_FIELDS,
    STRATEGY_SEARCH_FIELDS,
    get_allowed_ai_search_fields,
)


ALL_SEARCH_DIMENSIONS = {"basic", "augmentation", "loss", "strategy", "model_module"}
NON_BASIC_SEARCH_DIMENSIONS = {"augmentation", "loss", "strategy", "model_module"}


def get_default_run_policy() -> RunPolicy:
    """Return the default run policy used across the backend."""
    return RunPolicy()


def get_default_ranking_policy() -> RankingPolicy:
    """Return the default experiment ranking policy."""
    return RankingPolicy()


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
    source_experiment = _resolve_source_experiment(experiment, history_index=None)
    if source_experiment is not None:
        current_params = _extract_experiment_search_values(experiment)
        source_params = _extract_experiment_search_values(source_experiment)
        return {
            field_name
            for field_name, current_value in current_params.items()
            if source_params.get(field_name) != current_value
        }

    proposal_payload = experiment.get("proposal") or {}
    changes = proposal_payload.get("changes") or {}
    return {
        field_name
        for field_name, value in changes.items()
        if value is not None
    }


def _extract_experiment_params(experiment: dict[str, Any]) -> dict[str, Any]:
    """Return the structured params payload from one history item."""
    if isinstance(experiment.get("params"), dict):
        return experiment["params"]
    config_payload = experiment.get("config") or {}
    params_payload = config_payload.get("params")
    return params_payload if isinstance(params_payload, dict) else {}


def _extract_experiment_search_values(experiment: dict[str, Any]) -> dict[str, Any]:
    """Return one normalized search payload from recipes or legacy params."""
    if isinstance(experiment.get("train_hyp"), dict):
        train_hyp_payload = experiment["train_hyp"]
        model_recipe_payload = experiment.get("model_recipe") or {}
    else:
        config_payload = experiment.get("config") or {}
        train_hyp_payload = config_payload.get("train_hyp") or {}
        model_recipe_payload = config_payload.get("model_recipe") or {}

    if isinstance(train_hyp_payload, dict) and train_hyp_payload:
        augmentation_payload = train_hyp_payload.get("augmentation") or {}
        loss_payload = train_hyp_payload.get("loss") or {}
        modules_payload = model_recipe_payload.get("modules") or {}
        legacy_head_payload = model_recipe_payload.get("head")
        head_config_payload = model_recipe_payload.get("head_config") or (
            legacy_head_payload if isinstance(legacy_head_payload, dict) else {}
        )
        components_payload = model_recipe_payload.get("components") or {}
        backbone_component_payload = components_payload.get("backbone") or {}
        neck_component_payload = components_payload.get("neck") or {}
        head_component_payload = components_payload.get("head") or {}
        flattened_payload = {
            "optimizer": train_hyp_payload.get("optimizer"),
            "learning_rate": train_hyp_payload.get("lr0"),
            "batch_size": train_hyp_payload.get("batch_size"),
            "weight_decay": train_hyp_payload.get("weight_decay"),
            "scheduler": train_hyp_payload.get("scheduler"),
            "label_smoothing": train_hyp_payload.get("label_smoothing"),
            "image_size": train_hyp_payload.get("image_size"),
            "augmentation_policy": augmentation_payload.get("policy"),
            "mixup_alpha": augmentation_payload.get("mixup"),
            "cutmix_alpha": augmentation_payload.get("cutmix"),
            "random_erasing_prob": augmentation_payload.get("random_erasing"),
            "loss_name": loss_payload.get("name"),
            "focal_gamma": train_hyp_payload.get("fl_gamma"),
            "aux_logits": modules_payload.get("aux_logits"),
            "width_multiple": model_recipe_payload.get("width_multiple"),
            "pooling_type": head_config_payload.get("pooling_type"),
            "classifier_dropout": head_config_payload.get("classifier_dropout"),
            "backbone_name": backbone_component_payload.get("name"),
            "neck_name": neck_component_payload.get("name"),
            "head_name": head_component_payload.get("name"),
        }
        return {
            field_name: value
            for field_name, value in flattened_payload.items()
            if value is not None
        }

    return _flatten_search_params(_extract_experiment_params(experiment))


def _flatten_search_params(params_payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested params into the search field namespace."""
    augmentation_params = params_payload.get("augmentation_params") or {}
    loss_params = params_payload.get("loss_params") or {}
    flattened_payload = {
        "optimizer": params_payload.get("optimizer"),
        "learning_rate": params_payload.get("learning_rate"),
        "batch_size": params_payload.get("batch_size"),
        "weight_decay": params_payload.get("weight_decay"),
        "scheduler": params_payload.get("scheduler"),
        "label_smoothing": params_payload.get("label_smoothing"),
        "image_size": params_payload.get("image_size"),
        "augmentation_policy": params_payload.get("augmentation_policy"),
        "mixup_alpha": augmentation_params.get("mixup_alpha"),
        "cutmix_alpha": augmentation_params.get("cutmix_alpha"),
        "random_erasing_prob": augmentation_params.get("random_erasing_prob"),
        "loss_name": params_payload.get("loss_name"),
        "focal_gamma": loss_params.get("focal_gamma"),
        "aux_logits": params_payload.get("aux_logits"),
    }
    return {
        field_name: value
        for field_name, value in flattened_payload.items()
        if value is not None
    }


def _build_history_index(experiment_history: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return one lookup map for experiment history items."""
    return {
        experiment["id"]: experiment
        for experiment in experiment_history
        if isinstance(experiment.get("id"), str)
    }


def _resolve_source_experiment(
    experiment: dict[str, Any],
    history_index: dict[str, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Return the primary source experiment for one history item, if any."""
    if history_index is None:
        return None
    proposal_payload = experiment.get("proposal") or {}
    based_on_experiment_ids = proposal_payload.get("based_on_experiment_ids") or []
    for experiment_id in based_on_experiment_ids:
        source_experiment = history_index.get(experiment_id)
        if source_experiment is not None:
            return source_experiment
    return None


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
    if field_name in MODEL_MODULE_SEARCH_FIELDS:
        return "model_module"
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
    history_index = _build_history_index(experiment_history)
    latest_index = len(stagnation_history) - 1
    if latest_index < 0:
        return set()

    blocked_fields: set[str] = set()
    all_fields = {
        field_name
        for experiment in stagnation_history
        for field_name in _get_effective_history_change_fields_with_index(experiment, history_index)
    }
    for field_name in all_fields:
        streak_length = 0
        cooldown_anchor_index: int | None = None
        for index, experiment in enumerate(stagnation_history):
            changed_fields = _get_effective_history_change_fields_with_index(experiment, history_index)
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
    history_index = _build_history_index(experiment_history)
    if len(stagnation_history) < effective_policy.stagnation_rounds_for_dimension_switch:
        return set()

    recent_history = stagnation_history[-effective_policy.stagnation_rounds_for_dimension_switch :]
    recent_dimension_sets = [
        get_change_dimensions(_get_effective_history_change_fields_with_index(experiment, history_index))
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


def _get_effective_history_change_fields_with_index(
    experiment: dict[str, Any],
    history_index: dict[str, dict[str, Any]],
) -> set[str]:
    """Return changed fields using both source-diffing and proposal fallbacks."""
    source_experiment = _resolve_source_experiment(experiment, history_index)
    if source_experiment is not None:
        current_params = _extract_experiment_search_values(experiment)
        source_params = _extract_experiment_search_values(source_experiment)
        return {
            field_name
            for field_name, current_value in current_params.items()
            if source_params.get(field_name) != current_value
        }
    return _get_effective_history_change_fields(experiment)


def get_available_dimensions(search_policy: SearchPolicy | dict[str, Any] | None) -> set[str]:
    """Return the search dimensions currently open to the AI."""
    policy = search_policy
    if isinstance(search_policy, dict):
        policy = SearchPolicy.model_validate(search_policy)
    allowed_fields = get_allowed_ai_search_fields(policy)
    return get_change_dimensions(allowed_fields)


def get_preferred_fields(
    allowed_fields: set[str],
    *,
    blocked_fields: set[str] | None = None,
    discouraged_dimensions: set[str] | None = None,
    prefer_non_basic: bool = False,
) -> tuple[set[str], list[str]]:
    """Return a soft-preference field set without collapsing the feasible space."""
    preferred_fields = set(allowed_fields)
    preference_notes: list[str] = []
    blocked_fields = blocked_fields or set()
    discouraged_dimensions = discouraged_dimensions or set()

    if prefer_non_basic:
        non_basic_fields = {
            field_name
            for field_name in preferred_fields
            if get_field_dimension(field_name) in NON_BASIC_SEARCH_DIMENSIONS
        }
        if non_basic_fields:
            preferred_fields = non_basic_fields
            preference_notes.append("prefer_non_basic")

    if blocked_fields:
        non_blocked_fields = preferred_fields - blocked_fields
        if non_blocked_fields:
            preferred_fields = non_blocked_fields
            preference_notes.append("avoid_recent_failed_fields")

    if discouraged_dimensions:
        switched_fields = {
            field_name
            for field_name in preferred_fields
            if get_field_dimension(field_name) not in discouraged_dimensions
        }
        if switched_fields:
            preferred_fields = switched_fields
            preference_notes.append("prefer_dimension_switch")

    return preferred_fields or set(allowed_fields), preference_notes


def get_dimension_attempt_count(
    experiment_history: list[dict[str, Any]],
) -> dict[str, int]:
    """Return how many successful experiments have explored each dimension."""
    history_index = _build_history_index(experiment_history)
    attempt_count = {
        dimension: 0
        for dimension in ALL_SEARCH_DIMENSIONS
    }
    for experiment in experiment_history:
        if experiment.get("status") != "success":
            continue
        changed_fields = _get_effective_history_change_fields_with_index(experiment, history_index)
        for dimension in get_change_dimensions(changed_fields):
            attempt_count[dimension] += 1
    return attempt_count


def should_stop_after_dimension_coverage(
    experiment_history: list[dict[str, Any]],
    search_policy: SearchPolicy | dict[str, Any] | None,
    *,
    policy: RunPolicy | None = None,
) -> tuple[bool, str]:
    """Return whether auto-train should stop after exhausting the allowed dimensions."""
    effective_policy = policy or get_default_run_policy()
    available_dimensions = get_available_dimensions(search_policy)
    if not available_dimensions:
        return False, "No AI-search dimensions are enabled for the current run."

    dimension_attempt_count = get_dimension_attempt_count(experiment_history)
    explored_dimensions = {
        dimension
        for dimension, attempt_count in dimension_attempt_count.items()
        if attempt_count > 0
    }
    if not available_dimensions.issubset(explored_dimensions):
        missing_dimensions = sorted(available_dimensions - explored_dimensions)
        return False, f"Still missing explored dimensions: {', '.join(missing_dimensions)}."

    underexplored_dimensions = sorted(
        dimension
        for dimension in available_dimensions
        if dimension_attempt_count.get(dimension, 0) < effective_policy.auto_train_min_successful_attempts_per_dimension
    )
    if underexplored_dimensions:
        return (
            False,
            "Some dimensions have not reached the minimum successful attempt budget: "
            f"{', '.join(underexplored_dimensions)}.",
        )

    stagnation_rounds = count_consecutive_stagnation_rounds(experiment_history)
    if stagnation_rounds < effective_policy.auto_train_early_stop_stagnation_rounds:
        return (
            False,
            "Dimension coverage is complete, but the recent stagnation window is still below the stop threshold.",
        )

    return (
        True,
        "Stopped by run policy: all enabled search dimensions reached the minimum attempt budget "
        f"and the run has stalled for {stagnation_rounds} rounds.",
    )


def require_non_basic_change_after_warmup_rounds(
    current_round: int,
    *,
    policy: RunPolicy | None = None,
) -> bool:
    """Return whether the current round should prioritize augmentation/loss/strategy changes."""
    effective_policy = policy or get_default_run_policy()
    return current_round > effective_policy.auto_train_non_basic_change_after_round


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


def _get_ranking_metric_value(experiment: ExperimentModel, metric_name: str) -> float | None:
    """Return one comparable ranking metric from the persisted experiment."""
    result_payload = experiment.result or {}
    metrics_payload = result_payload.get("metrics") or {}
    resource_payload = result_payload.get("resource") or {}

    if metric_name in {"training_seconds", "latency_ms", "parameter_count_million"}:
        metric_value = resource_payload.get(metric_name)
    else:
        metric_value = metrics_payload.get(metric_name)

    if isinstance(metric_value, (int, float)):
        return float(metric_value)
    return None


def _get_experiment_image_size(experiment: ExperimentModel) -> int | None:
    """Return image_size from one persisted experiment config."""
    config_payload = experiment.experiment_config or {}
    train_hyp_payload = config_payload.get("train_hyp") or {}
    image_size = train_hyp_payload.get("image_size")
    if not isinstance(image_size, int):
        params_payload = config_payload.get("params") or {}
        image_size = params_payload.get("image_size")
    return int(image_size) if isinstance(image_size, int) else None


def get_experiment_ranking_policy(experiment: ExperimentModel | None) -> RankingPolicy:
    """Return the effective ranking policy for one persisted experiment."""
    if experiment is None:
        return get_default_ranking_policy()
    config_payload = experiment.experiment_config or {}
    return RankingPolicy.model_validate(config_payload.get("ranking_policy") or {})


def _compute_metric_improvement(
    candidate_value: float,
    incumbent_value: float,
    *,
    metric_mode: RankingMetricMode,
) -> float:
    """Return positive values when the candidate improved under the given metric mode."""
    if metric_mode == "max":
        return candidate_value - incumbent_value
    return incumbent_value - candidate_value


def _passes_cost_gate(candidate: ExperimentModel, ranking_policy: RankingPolicy) -> tuple[bool, str | None]:
    """Return whether the candidate passes all configured cost gates."""
    candidate_image_size = _get_experiment_image_size(candidate)
    if ranking_policy.max_image_size is not None and candidate_image_size is not None:
        if candidate_image_size > ranking_policy.max_image_size:
            return (
                False,
                f"image_size {candidate_image_size} exceeds max_image_size {ranking_policy.max_image_size}",
            )
    return True, None


def _evaluate_tie_breaker(
    candidate: ExperimentModel,
    incumbent: ExperimentModel,
    ranking_policy: RankingPolicy,
) -> tuple[bool, str]:
    """Return whether the candidate wins on the configured tie-breaker."""
    candidate_tie_breaker = _get_ranking_metric_value(candidate, ranking_policy.tie_breaker_metric)
    incumbent_tie_breaker = _get_ranking_metric_value(incumbent, ranking_policy.tie_breaker_metric)

    if candidate_tie_breaker is not None and incumbent_tie_breaker is not None:
        tie_breaker_improvement = _compute_metric_improvement(
            candidate_tie_breaker,
            incumbent_tie_breaker,
            metric_mode=ranking_policy.tie_breaker_mode,
        )
        if tie_breaker_improvement >= ranking_policy.min_tie_breaker_metric_improvement:
            return (
                True,
                "Promoted by ranking policy: "
                f"{ranking_policy.tie_breaker_metric} improved from "
                f"{format_metric_value(incumbent_tie_breaker)} to "
                f"{format_metric_value(candidate_tie_breaker)} at comparable primary metric.",
            )
        return (
            False,
            "Discarded by ranking policy: comparable primary metric but "
            f"{ranking_policy.tie_breaker_metric} did not improve by at least "
            f"{ranking_policy.min_tie_breaker_metric_improvement:.3f}.",
        )

    if incumbent_tie_breaker is None and candidate_tie_breaker is not None:
        return (
            True,
            "Promoted by ranking policy: candidate reports "
            f"{ranking_policy.tie_breaker_metric} while the incumbent does not.",
        )

    if incumbent_tie_breaker is not None and candidate_tie_breaker is None:
        return (
            False,
            "Discarded by ranking policy: incumbent reports "
            f"{ranking_policy.tie_breaker_metric} but the candidate does not.",
        )

    return (
        False,
        "Discarded by ranking policy: neither candidate nor incumbent provides "
        f"{ranking_policy.tie_breaker_metric} for tie-breaking.",
    )


def evaluate_promotion(
    candidate: ExperimentModel,
    incumbent: ExperimentModel | None,
    *,
    ranking_policy: RankingPolicy | None = None,
) -> tuple[bool, str]:
    """Return whether the candidate should be promoted over the incumbent."""
    effective_ranking_policy = ranking_policy or get_default_ranking_policy()
    passes_cost_gate, cost_gate_reason = _passes_cost_gate(candidate, effective_ranking_policy)
    if not passes_cost_gate:
        return False, f"Discarded by ranking policy: {cost_gate_reason}."

    candidate_primary_metric = _get_ranking_metric_value(candidate, effective_ranking_policy.primary_metric)
    if incumbent is None:
        return True, "Promoted as the first successful experiment under the ranking policy."

    incumbent_primary_metric = _get_ranking_metric_value(incumbent, effective_ranking_policy.primary_metric)
    if incumbent_primary_metric is not None and candidate_primary_metric is not None:
        primary_metric_improvement = _compute_metric_improvement(
            candidate_primary_metric,
            incumbent_primary_metric,
            metric_mode=effective_ranking_policy.primary_metric_mode,
        )
        if primary_metric_improvement >= effective_ranking_policy.min_primary_metric_improvement:
            return (
                True,
                "Promoted by ranking policy: "
                f"{effective_ranking_policy.primary_metric} improved from "
                f"{format_metric_value(incumbent_primary_metric)} to "
                f"{format_metric_value(candidate_primary_metric)}.",
            )
        if abs(candidate_primary_metric - incumbent_primary_metric) <= effective_ranking_policy.primary_metric_parity_epsilon:
            return _evaluate_tie_breaker(candidate, incumbent, effective_ranking_policy)
        return (
            False,
            "Discarded by ranking policy: "
            f"{effective_ranking_policy.primary_metric} "
            f"({format_metric_value(candidate_primary_metric)}) did not beat current best "
            f"({format_metric_value(incumbent_primary_metric)}) by at least "
            f"{effective_ranking_policy.min_primary_metric_improvement:.3f}.",
        )

    if incumbent_primary_metric is None and candidate_primary_metric is not None:
        return (
            True,
            "Promoted by ranking policy: candidate reports "
            f"{effective_ranking_policy.primary_metric} while the incumbent does not.",
        )

    if incumbent_primary_metric is not None and candidate_primary_metric is None:
        return (
            False,
            "Discarded by ranking policy: incumbent reports "
            f"{effective_ranking_policy.primary_metric} but the candidate does not.",
        )

    return _evaluate_tie_breaker(candidate, incumbent, effective_ranking_policy)
