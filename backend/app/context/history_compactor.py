"""Compact proposal-stage history into structured summaries."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from app.schemas.prompt_context import (
    CurrentStageCompactedBucket,
    CurrentStageCompactedSummary,
    HistoryItemSummary,
    PastStageSummary,
)


CURRENT_STAGE_BUCKET_SIZE = 10


def _direction_label(history_item: HistoryItemSummary) -> str:
    if history_item.delta_from_source:
        parts = [f"{item.field}: {item.from_value} -> {item.to_value}" for item in history_item.delta_from_source]
        return ", ".join(parts)
    return "no parameter changes"


def _collect_failure_patterns(history_items: Iterable[HistoryItemSummary]) -> list[str]:
    failure_counter: Counter[str] = Counter()
    for history_item in history_items:
        if history_item.execution_error_summary:
            failure_counter[history_item.execution_error_summary] += 1
        elif history_item.decision in {"crash", "timeout"}:
            failure_counter[f"decision={history_item.decision}"] += 1
        elif history_item.result_snapshot.status == "failed":
            failure_counter["status=failed"] += 1
    return [f"{label} (x{count})" for label, count in failure_counter.most_common()]


def _compact_history_bucket(
    history_items: list[HistoryItemSummary],
    *,
    bucket_index: int,
) -> CurrentStageCompactedBucket:
    """Compact one bucket of older current-stage items."""
    effective_directions: list[str] = []
    ineffective_directions: list[str] = []
    for history_item in history_items:
        direction_label = _direction_label(history_item)
        if history_item.decision == "keep":
            effective_directions.append(direction_label)
        elif history_item.decision in {"discard", "crash", "timeout"} or history_item.result_snapshot.status == "failed":
            ineffective_directions.append(direction_label)
    return CurrentStageCompactedBucket(
        bucket_index=bucket_index,
        effective_directions=sorted(dict.fromkeys(effective_directions)),
        ineffective_directions=sorted(dict.fromkeys(ineffective_directions)),
        failure_patterns=_collect_failure_patterns(history_items),
        covered_experiment_ids=[history_item.experiment_id for history_item in history_items],
    )


def compact_current_stage_history(history_items: list[HistoryItemSummary]) -> CurrentStageCompactedSummary:
    """Compact older current-stage items into fixed-size bucket summaries."""
    buckets = [
        _compact_history_bucket(
            history_items[start_index : start_index + CURRENT_STAGE_BUCKET_SIZE],
            bucket_index=bucket_index,
        )
        for bucket_index, start_index in enumerate(range(0, len(history_items), CURRENT_STAGE_BUCKET_SIZE), start=1)
    ]
    return CurrentStageCompactedSummary(buckets=buckets)


def compact_past_stage_history(
    *,
    from_best_experiment_id: str,
    to_best_experiment_id: str,
    history_items: list[HistoryItemSummary],
) -> PastStageSummary:
    """Compact one historical best-to-best stage."""
    compacted = _compact_history_bucket(history_items, bucket_index=1)
    return PastStageSummary(
        from_best_experiment_id=from_best_experiment_id,
        to_best_experiment_id=to_best_experiment_id,
        effective_directions=compacted.effective_directions,
        ineffective_directions=compacted.ineffective_directions,
        failure_patterns=compacted.failure_patterns,
        covered_experiment_ids=compacted.covered_experiment_ids,
    )
