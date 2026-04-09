"""Run-level in-memory cache for proposal history context state."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Hashable

from app.context.context_mapper import build_history_item_summary
from app.context.history_compactor import CURRENT_STAGE_BUCKET_SIZE, compact_current_stage_history
from app.context.history_selector import _is_stage_anchor
from app.context.types import DEFAULT_STAGE_COMPACT_THRESHOLD, DEFAULT_STAGE_HISTORY_KEEP
from app.schemas.prompt_context import (
    CurrentStageCompactedSummary,
    HistoryItemSummary,
    PastStageSummary,
    ProposalHistoryContext,
)


@dataclass(slots=True)
class RunProposalContextCacheEntry:
    """Cached proposal context state for one run."""

    run_id: str
    history_signature: Hashable
    run_payload: dict[str, Any]
    history_item_count: int
    base_experiment_payload: dict[str, Any]
    source_experiment_payload: dict[str, Any]
    last_processed_experiment_id: str
    last_processed_experiment_created_at: datetime | None
    recent_history_queue: deque[HistoryItemSummary] = field(default_factory=deque)
    compacted_bucket_queue: deque = field(default_factory=deque)
    compacted_bucket_pending_queue: deque[HistoryItemSummary] = field(default_factory=deque)
    past_stage_summaries: list[PastStageSummary] = field(default_factory=list)
    current_stage_item_count: int = 0
    current_stage_effective_directions: set[str] = field(default_factory=set)
    current_stage_ineffective_directions: set[str] = field(default_factory=set)
    current_stage_failure_counter: Counter[str] = field(default_factory=Counter)
    current_stage_covered_experiment_ids: list[str] = field(default_factory=list)

    def to_history_context(self) -> ProposalHistoryContext:
        """Return one prompt-builder history context snapshot."""
        compacted_summary = None
        compacted_buckets = self._build_compacted_buckets_for_prompt()
        if compacted_buckets:
            compacted_summary = CurrentStageCompactedSummary(
                buckets=compacted_buckets,
            )
        return ProposalHistoryContext(
            base_experiment_payload=self.base_experiment_payload,
            source_experiment_payload=self.source_experiment_payload,
            recent_stage_history=list(self.recent_history_queue),
            current_stage_compacted_summary=compacted_summary,
            past_stage_summaries=list(self.past_stage_summaries),
        )

    def summarize_cache_state(self) -> dict[str, Any]:
        """Return one log-friendly summary of current cached history state."""
        compacted_buckets = self._build_compacted_buckets_for_prompt()
        compacted_items = sum(len(bucket.covered_experiment_ids) for bucket in compacted_buckets)
        return {
            "history_items": self.history_item_count,
            "recent_history_items": len(self.recent_history_queue),
            "compacted_bucket_count": len(compacted_buckets),
            "compacted_history_items": compacted_items,
            "past_stage_summary_count": len(self.past_stage_summaries),
            "source_experiment_id": self.source_experiment_payload.get("id"),
        }

    def build_source_constraints(self) -> dict[str, Any]:
        """Return the current source metadata used by proposal generation."""
        train_hyp_payload = self.source_experiment_payload.get("train_hyp") or {}
        if not train_hyp_payload and isinstance(self.source_experiment_payload.get("config"), dict):
            config_payload = self.source_experiment_payload.get("config") or {}
            train_hyp_payload = config_payload.get("train_hyp") or {}
        return {
            "experiment_id": self.source_experiment_payload.get("id"),
            "image_size": train_hyp_payload.get("image_size"),
        }

    def refresh_run_payload(self, *, best_experiment_id: str | None) -> None:
        """Refresh run payload fields that change across proposal requests."""
        self.run_payload["best_experiment_id"] = best_experiment_id
        self.run_payload["experiment_count"] = self.history_item_count

    def append_experiment_payload(
        self,
        experiment_payload: dict[str, Any],
        *,
        stage_history_keep: int = DEFAULT_STAGE_HISTORY_KEEP,
        stage_compact_threshold: int = DEFAULT_STAGE_COMPACT_THRESHOLD,
    ) -> None:
        """Advance this run state with one new experiment payload."""
        self.history_item_count += 1
        self.last_processed_experiment_id = str(experiment_payload.get("id"))
        created_at = experiment_payload.get("created_at")
        if isinstance(created_at, datetime):
            self.last_processed_experiment_created_at = created_at
        elif isinstance(created_at, str):
            self.last_processed_experiment_created_at = datetime.fromisoformat(created_at)
        else:
            self.last_processed_experiment_created_at = None
        if _is_stage_anchor(experiment_payload):
            self._finalize_current_stage(next_source_experiment_payload=experiment_payload)
            return
        history_item = build_history_item_summary(
            source_experiment_payload=self.source_experiment_payload,
            experiment_payload=experiment_payload,
        )
        self._record_current_stage_item(history_item)
        self.recent_history_queue.append(history_item)
        if self.current_stage_item_count <= stage_compact_threshold:
            return
        while len(self.recent_history_queue) > stage_history_keep:
            self._append_compacted_item(self.recent_history_queue.popleft())

    def _build_compacted_buckets_for_prompt(self) -> list:
        compacted_buckets = list(self.compacted_bucket_queue)
        if not self.compacted_bucket_pending_queue:
            return compacted_buckets
        partial_summary = compact_current_stage_history(list(self.compacted_bucket_pending_queue))
        if partial_summary.buckets:
            compacted_buckets.append(
                partial_summary.buckets[0].model_copy(update={"bucket_index": len(compacted_buckets) + 1})
            )
        return compacted_buckets

    def _record_current_stage_item(self, history_item: HistoryItemSummary) -> None:
        direction_label = _direction_label(history_item)
        if history_item.decision == "keep":
            self.current_stage_effective_directions.add(direction_label)
        elif history_item.decision in {"discard", "crash", "timeout"} or history_item.result_snapshot.status == "failed":
            self.current_stage_ineffective_directions.add(direction_label)
        failure_label = _failure_pattern_label(history_item)
        if failure_label is not None:
            self.current_stage_failure_counter[failure_label] += 1
        self.current_stage_covered_experiment_ids.append(history_item.experiment_id)
        self.current_stage_item_count += 1

    def _append_compacted_item(self, history_item: HistoryItemSummary) -> None:
        self.compacted_bucket_pending_queue.append(history_item)
        if len(self.compacted_bucket_pending_queue) < CURRENT_STAGE_BUCKET_SIZE:
            return
        finalized_bucket_summary = compact_current_stage_history(list(self.compacted_bucket_pending_queue))
        if finalized_bucket_summary.buckets:
            self.compacted_bucket_queue.append(
                finalized_bucket_summary.buckets[0].model_copy(update={"bucket_index": len(self.compacted_bucket_queue) + 1})
            )
        self.compacted_bucket_pending_queue.clear()

    def _finalize_current_stage(self, *, next_source_experiment_payload: dict[str, Any]) -> None:
        self.past_stage_summaries.append(
            PastStageSummary(
                from_best_experiment_id=str(self.source_experiment_payload.get("id")),
                to_best_experiment_id=str(next_source_experiment_payload.get("id")),
                effective_directions=sorted(self.current_stage_effective_directions),
                ineffective_directions=sorted(self.current_stage_ineffective_directions),
                failure_patterns=_format_failure_patterns(self.current_stage_failure_counter),
                covered_experiment_ids=list(self.current_stage_covered_experiment_ids),
            )
        )
        self.source_experiment_payload = next_source_experiment_payload
        self.recent_history_queue.clear()
        self.compacted_bucket_queue.clear()
        self.compacted_bucket_pending_queue.clear()
        self.current_stage_item_count = 0
        self.current_stage_effective_directions.clear()
        self.current_stage_ineffective_directions.clear()
        self.current_stage_failure_counter.clear()
        self.current_stage_covered_experiment_ids.clear()


_PROPOSAL_CONTEXT_CACHE_LOCK = Lock()
_RUN_PROPOSAL_CONTEXT_CACHE: dict[str, RunProposalContextCacheEntry] = {}


def _direction_label(history_item: HistoryItemSummary) -> str:
    if history_item.delta_from_source:
        parts = [f"{item.field}: {item.from_value} -> {item.to_value}" for item in history_item.delta_from_source]
        return ", ".join(parts)
    return "no parameter changes"


def _failure_pattern_label(history_item: HistoryItemSummary) -> str | None:
    if history_item.execution_error_summary:
        return history_item.execution_error_summary
    if history_item.decision in {"crash", "timeout"}:
        return f"decision={history_item.decision}"
    if history_item.result_snapshot.status == "failed":
        return "status=failed"
    return None


def _format_failure_patterns(failure_counter: Counter[str]) -> list[str]:
    return [f"{label} (x{count})" for label, count in failure_counter.most_common()]


def clear_run_proposal_context_cache(run_id: str | None = None) -> None:
    """Clear one run cache entry or the full proposal context cache."""
    with _PROPOSAL_CONTEXT_CACHE_LOCK:
        if run_id is None:
            _RUN_PROPOSAL_CONTEXT_CACHE.clear()
            return
        _RUN_PROPOSAL_CONTEXT_CACHE.pop(run_id, None)


def get_run_proposal_context_cache_entry(run_id: str) -> RunProposalContextCacheEntry | None:
    """Return the cached proposal context state for one run if present."""
    with _PROPOSAL_CONTEXT_CACHE_LOCK:
        return _RUN_PROPOSAL_CONTEXT_CACHE.get(run_id)


def set_run_proposal_context_cache_entry(entry: RunProposalContextCacheEntry) -> None:
    """Persist one run proposal context state in the process-local cache."""
    with _PROPOSAL_CONTEXT_CACHE_LOCK:
        _RUN_PROPOSAL_CONTEXT_CACHE[entry.run_id] = entry
