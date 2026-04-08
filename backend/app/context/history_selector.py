"""Select baseline, source, and stage segments for proposal prompting."""

from __future__ import annotations

from typing import Any


def _experiment_id(value: Any) -> str:
    """Normalize experiment ids for cross-payload comparisons."""
    return str(value)


def _is_stage_anchor(experiment_payload: dict[str, Any]) -> bool:
    """Return whether this experiment should act as a best-stage anchor."""
    if experiment_payload.get("decision") == "keep":
        return True
    return bool(experiment_payload.get("is_best_so_far"))


def resolve_baseline_experiment(experiment_history: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the baseline experiment from the ordered history."""
    if not experiment_history:
        raise ValueError("experiment_history must not be empty")
    return experiment_history[0]


def resolve_source_experiment(
    experiment_history: list[dict[str, Any]],
    best_experiment_id: str | None,
) -> dict[str, Any]:
    """Return the current source experiment for proposal generation."""
    if not experiment_history:
        raise ValueError("experiment_history must not be empty")
    if best_experiment_id:
        normalized_best_experiment_id = _experiment_id(best_experiment_id)
        for experiment_payload in experiment_history:
            if _experiment_id(experiment_payload.get("id")) == normalized_best_experiment_id:
                return experiment_payload
    return experiment_history[-1]


def select_current_stage_history(
    experiment_history: list[dict[str, Any]],
    source_experiment_id: str,
) -> list[dict[str, Any]]:
    """Return experiments created after the current source experiment."""
    normalized_source_experiment_id = _experiment_id(source_experiment_id)
    source_index = next(
        (
            index
            for index, experiment_payload in enumerate(experiment_history)
            if _experiment_id(experiment_payload.get("id")) == normalized_source_experiment_id
        ),
        None,
    )
    if source_index is None:
        raise ValueError("source experiment is missing from experiment history")
    return experiment_history[source_index + 1 :]


def select_past_stage_ranges(
    experiment_history: list[dict[str, Any]],
    source_experiment_id: str,
) -> list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]]:
    """Return historical best-to-best stages before the current source."""
    normalized_source_experiment_id = _experiment_id(source_experiment_id)
    source_index = next(
        (
            index
            for index, experiment_payload in enumerate(experiment_history)
            if _experiment_id(experiment_payload.get("id")) == normalized_source_experiment_id
        ),
        None,
    )
    if source_index is None:
        raise ValueError("source experiment is missing from experiment history")
    history_before_source = experiment_history[: source_index + 1]
    stage_anchors = [experiment_payload for experiment_payload in history_before_source if _is_stage_anchor(experiment_payload)]
    if len(stage_anchors) < 2:
        return []
    stage_ranges: list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = []
    for previous_anchor, next_anchor in zip(stage_anchors[:-1], stage_anchors[1:]):
        previous_index = history_before_source.index(previous_anchor)
        next_index = history_before_source.index(next_anchor)
        stage_ranges.append(
            (
                previous_anchor,
                next_anchor,
                history_before_source[previous_index + 1 : next_index],
            )
        )
        if _experiment_id(next_anchor.get("id")) == normalized_source_experiment_id:
            break
    return stage_ranges
