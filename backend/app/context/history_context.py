"""Build reusable proposal history context state from ordered experiment history."""

from __future__ import annotations

from app.context.context_mapper import build_history_item_summary
from app.context.history_compactor import compact_current_stage_history, compact_past_stage_history
from app.context.history_selector import (
    resolve_baseline_experiment,
    resolve_source_experiment,
    select_current_stage_history,
    select_past_stage_ranges,
)
from app.context.types import DEFAULT_STAGE_COMPACT_THRESHOLD, DEFAULT_STAGE_HISTORY_KEEP
from app.schemas.prompt_context import ProposalHistoryContext


def build_proposal_history_context(
    *,
    run_payload: dict,
    experiment_history: list[dict],
    stage_history_keep: int = DEFAULT_STAGE_HISTORY_KEEP,
    stage_compact_threshold: int = DEFAULT_STAGE_COMPACT_THRESHOLD,
) -> ProposalHistoryContext:
    """Build selected and compacted run history for one proposal request."""
    if not experiment_history:
        raise ValueError("experiment_history must not be empty")
    base_experiment = resolve_baseline_experiment(experiment_history)
    source_experiment = resolve_source_experiment(experiment_history, run_payload.get("best_experiment_id"))
    current_stage_experiments = select_current_stage_history(experiment_history, str(source_experiment.get("id")))
    current_stage_history = [
        build_history_item_summary(
            source_experiment_payload=source_experiment,
            experiment_payload=experiment_payload,
        )
        for experiment_payload in current_stage_experiments
    ]
    compacted_current_stage = None
    visible_stage_history = current_stage_history
    if len(current_stage_history) > stage_compact_threshold:
        older_stage_history = current_stage_history[:-stage_history_keep]
        visible_stage_history = current_stage_history[-stage_history_keep:]
        compacted_current_stage = compact_current_stage_history(older_stage_history)
    stage_summaries = []
    for previous_best, next_best, stage_experiments in select_past_stage_ranges(
        experiment_history,
        str(source_experiment.get("id")),
    ):
        stage_history_items = [
            build_history_item_summary(
                source_experiment_payload=previous_best,
                experiment_payload=experiment_payload,
            )
            for experiment_payload in stage_experiments
        ]
        stage_summaries.append(
            compact_past_stage_history(
                from_best_experiment_id=str(previous_best.get("id")),
                to_best_experiment_id=str(next_best.get("id")),
                history_items=stage_history_items,
            )
        )
    return ProposalHistoryContext(
        base_experiment_payload=base_experiment,
        source_experiment_payload=source_experiment,
        recent_stage_history=visible_stage_history,
        current_stage_compacted_summary=compacted_current_stage,
        past_stage_summaries=stage_summaries,
    )
