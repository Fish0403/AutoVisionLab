"""Run-level in-memory cache for proposal history context state."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Hashable

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
    experiment_history_snapshot: list[dict[str, Any]]
    source_constraints: dict[str, Any]
    base_experiment_payload: dict[str, Any]
    source_experiment_payload: dict[str, Any]
    recent_history_queue: deque[HistoryItemSummary] = field(default_factory=deque)
    compacted_bucket_queue: deque = field(default_factory=deque)
    past_stage_summaries: list[PastStageSummary] = field(default_factory=list)

    def to_history_context(self) -> ProposalHistoryContext:
        """Return one prompt-builder history context snapshot."""
        compacted_summary = None
        if self.compacted_bucket_queue:
            compacted_summary = CurrentStageCompactedSummary(
                buckets=list(self.compacted_bucket_queue),
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
        compacted_buckets = list(self.compacted_bucket_queue)
        compacted_items = sum(len(bucket.covered_experiment_ids) for bucket in compacted_buckets)
        return {
            "history_items": len(self.experiment_history_snapshot),
            "recent_history_items": len(self.recent_history_queue),
            "compacted_bucket_count": len(compacted_buckets),
            "compacted_history_items": compacted_items,
            "past_stage_summary_count": len(self.past_stage_summaries),
            "source_experiment_id": self.source_experiment_payload.get("id"),
        }


@dataclass(slots=True)
class RunProposalContextCacheLookup:
    """One cache lookup result plus hit/miss metadata."""

    entry: RunProposalContextCacheEntry
    cache_status: str


_PROPOSAL_CONTEXT_CACHE_LOCK = Lock()
_RUN_PROPOSAL_CONTEXT_CACHE: dict[str, RunProposalContextCacheEntry] = {}


def clear_run_proposal_context_cache(run_id: str | None = None) -> None:
    """Clear one run cache entry or the full proposal context cache."""
    with _PROPOSAL_CONTEXT_CACHE_LOCK:
        if run_id is None:
            _RUN_PROPOSAL_CONTEXT_CACHE.clear()
            return
        _RUN_PROPOSAL_CONTEXT_CACHE.pop(run_id, None)


def get_or_build_run_proposal_context_cache_entry(
    *,
    run_id: str,
    history_signature: Hashable,
    builder: Callable[[], RunProposalContextCacheEntry],
) -> RunProposalContextCacheLookup:
    """Return one cached run entry when signatures match, or rebuild it lazily."""
    with _PROPOSAL_CONTEXT_CACHE_LOCK:
        cached_entry = _RUN_PROPOSAL_CONTEXT_CACHE.get(run_id)
        if cached_entry is not None and cached_entry.history_signature == history_signature:
            return RunProposalContextCacheLookup(entry=cached_entry, cache_status="hit")
    rebuilt_entry = builder()
    with _PROPOSAL_CONTEXT_CACHE_LOCK:
        cached_entry = _RUN_PROPOSAL_CONTEXT_CACHE.get(run_id)
        if cached_entry is not None and cached_entry.history_signature == history_signature:
            return RunProposalContextCacheLookup(entry=cached_entry, cache_status="hit_after_race")
        cache_status = "miss"
        if cached_entry is not None:
            cache_status = "rebuild"
        _RUN_PROPOSAL_CONTEXT_CACHE[run_id] = rebuilt_entry
        return RunProposalContextCacheLookup(entry=rebuilt_entry, cache_status=cache_status)
