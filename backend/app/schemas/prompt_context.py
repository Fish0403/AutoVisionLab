"""Internal schemas for proposal context assembly."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


PromptRole = Literal["system", "user"]
PromptBlockName = Literal[
    "system_prompt",
    "output_schema_prompt",
    "policy_prompt",
    "base_prompt",
    "source_prompt",
    "stage_history_prompt",
    "current_stage_compacted_prompt",
    "past_stage_summaries_prompt",
    "retry_prompt",
]
ChangeType = Literal["increase", "decrease", "switch", "toggle"]


class PromptBlock(BaseModel):
    """One prompt block before final rendering."""

    name: PromptBlockName
    role: PromptRole
    payload: Any
    render_priority: int


class MetricSnapshot(BaseModel):
    """Result metrics that can influence experiment ranking."""

    top1_acc: float | None = None
    val_loss: float | None = None
    train_loss: float | None = None
    best_epoch: int | None = None
    latency_ms: float | None = None
    parameter_count_million: float | None = None


class ResourceSnapshot(BaseModel):
    """Execution cost signals for one experiment."""

    training_seconds: int | None = None
    gpu_memory_mb: int | None = None


class ResultSnapshot(BaseModel):
    """Compact result view exposed to proposal prompting."""

    status: str | None = None
    metrics: MetricSnapshot = Field(default_factory=MetricSnapshot)
    resource: ResourceSnapshot = Field(default_factory=ResourceSnapshot)


class ConfigDeltaItem(BaseModel):
    """One structured parameter delta item."""

    field: str
    from_value: Any = None
    to_value: Any = None
    change_type: ChangeType


class BasePromptPayload(BaseModel):
    """The full baseline state used as the starting point."""

    experiment_id: str
    params: dict[str, Any] = Field(default_factory=dict)
    train_hyp: dict[str, Any] = Field(default_factory=dict)
    model_recipe: dict[str, Any] = Field(default_factory=dict)
    result_snapshot: ResultSnapshot = Field(default_factory=ResultSnapshot)
    summary: str | None = None


class SourcePromptPayload(BaseModel):
    """The current best/source state relative to the baseline."""

    experiment_id: str
    based_on_experiment_id: str
    delta_from_base: list[ConfigDeltaItem] = Field(default_factory=list)
    result_snapshot: ResultSnapshot = Field(default_factory=ResultSnapshot)
    is_best: bool = False
    summary: str | None = None


class HistoryItemSummary(BaseModel):
    """One current-stage attempt relative to the source."""

    experiment_id: str
    based_on_source_experiment_id: str
    delta_from_source: list[ConfigDeltaItem] = Field(default_factory=list)
    decision: str | None = None
    result_snapshot: ResultSnapshot = Field(default_factory=ResultSnapshot)
    outcome_summary: str | None = None
    execution_error_summary: str | None = None


class CurrentStageCompactedBucket(BaseModel):
    """Compacted summary for one bucket of older current-stage attempts."""

    bucket_index: int
    effective_directions: list[str] = Field(default_factory=list)
    ineffective_directions: list[str] = Field(default_factory=list)
    failure_patterns: list[str] = Field(default_factory=list)
    covered_experiment_ids: list[str] = Field(default_factory=list)


class CurrentStageCompactedSummary(BaseModel):
    """Compacted summaries for older attempts inside the current stage."""

    buckets: list[CurrentStageCompactedBucket] = Field(default_factory=list)


class PastStageSummary(BaseModel):
    """Compacted summary for one historical best-to-best stage."""

    from_best_experiment_id: str
    to_best_experiment_id: str
    effective_directions: list[str] = Field(default_factory=list)
    ineffective_directions: list[str] = Field(default_factory=list)
    failure_patterns: list[str] = Field(default_factory=list)
    covered_experiment_ids: list[str] = Field(default_factory=list)


class RetryPromptPayload(BaseModel):
    """Rejection feedback for the next proposal retry."""

    proposal_rejection_reason: str
    invalid_fields: list[str] = Field(default_factory=list)
    retry_guidance: str | None = None


class ProposalPromptContext(BaseModel):
    """Complete proposal prompt context before rendering."""

    base_prompt: BasePromptPayload
    source_prompt: SourcePromptPayload
    stage_history_prompt: list[HistoryItemSummary] = Field(default_factory=list)
    current_stage_compacted_prompt: CurrentStageCompactedSummary | None = None
    past_stage_summaries_prompt: list[PastStageSummary] = Field(default_factory=list)
    retry_prompt: RetryPromptPayload | None = None


class ProposalHistoryContext(BaseModel):
    """Selected and compacted history state used to build proposal prompts."""

    base_experiment_payload: dict[str, Any]
    source_experiment_payload: dict[str, Any]
    recent_stage_history: list[HistoryItemSummary] = Field(default_factory=list)
    current_stage_compacted_summary: CurrentStageCompactedSummary | None = None
    past_stage_summaries: list[PastStageSummary] = Field(default_factory=list)


class PromptBundle(BaseModel):
    """Rendered prompt strings plus block metadata."""

    system_prompt: str
    user_prompt: str
    blocks: list[PromptBlock] = Field(default_factory=list)
    prompt_chars: int = 0
    prompt_tokens_estimate: int = 0
