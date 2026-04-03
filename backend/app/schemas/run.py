"""Run-facing API schemas."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.common import PointMetric, RunStatus
from app.schemas.experiment import ExperimentSummary
from app.schemas.parameter_space import EditableParameterSpace, ExperimentConfig


class RunCreateRequest(BaseModel):
    """Request payload for creating a run."""

    name: str
    dataset: str
    model_name: str
    base_config: ExperimentConfig
    notes: str | None = None


class RunListItem(BaseModel):
    """Compact run row."""

    id: str
    name: str
    dataset: str
    model_name: str
    status: RunStatus
    experiment_count: int


class RunDetailResponse(BaseModel):
    """Full run payload for run detail page."""

    id: str
    name: str
    dataset: str
    model_name: str
    status: RunStatus
    notes: str | None = None
    best_experiment_id: str | None = None
    experiments: list[ExperimentSummary]


class RunSummaryResponse(BaseModel):
    """Compact run research summary with one active best experiment."""

    run_id: str
    best_experiment_id: str | None = None
    keep_count: int = 0
    discard_count: int = 0
    crash_count: int = 0
    timeout_count: int = 0


class RunMetricsResponse(BaseModel):
    """Trend chart payload for one run."""

    run_id: str
    metric_name: str
    available_metrics: list[str]
    points: list[PointMetric]


class AutoTrainStartRequest(BaseModel):
    """Request payload for starting one background auto-train task."""

    title: str | None = None
    run_id: str | None = None
    run_name: str
    dataset: str
    model_name: str
    policy_preset: str | None = None
    source_task_type: Literal["model_compare"] | None = None
    source_task_id: str | None = None
    source_task_title: str | None = None
    source_model_name: str | None = None
    config: ExperimentConfig
    parameter_space: EditableParameterSpace


class AutoTrainTaskResponse(BaseModel):
    """Background auto-train task snapshot."""

    task_id: str
    title: str | None = None
    status: str
    dataset: str | None = None
    model_name: str | None = None
    policy_preset: str | None = None
    search_scope_summary: str | None = None
    run_id: str | None = None
    source_task_type: Literal["model_compare"] | None = None
    source_task_id: str | None = None
    source_task_title: str | None = None
    source_model_name: str | None = None
    current_round: int = 0
    elapsed_seconds: float = 0.0
    current_experiment_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    activity_message: str | None = None
    proposal_warning: str | None = None
    dataset_summary: str | None = None
    training_image_size: int | None = None
    ai_model_name: str | None = None
    logs: list[str]
    summary: dict[str, Any] | None = None
    error: str | None = None
    stop_requested: bool = False
    stop_reason: str | None = None
    latest_prompt_tokens_estimate: int | None = None
    estimated_prompt_tokens_total: int = 0
    latest_prompt_history_items: int | None = None
    latest_provider_prompt_tokens: int | None = None
    latest_provider_completion_tokens: int | None = None
    latest_provider_total_tokens: int | None = None
    provider_prompt_tokens_total: int = 0
    provider_completion_tokens_total: int = 0
    provider_total_tokens_total: int = 0


class ModelCompareStartRequest(BaseModel):
    """Request payload for starting one cross-model compare task."""

    title: str | None = None
    dataset: str
    config: ExperimentConfig
    candidate_models: list[str] | None = None


class ModelCompareCandidateResult(BaseModel):
    """One candidate result inside the cross-model compare summary."""

    model_name: str
    run_id: str | None = None
    baseline_experiment_id: str | None = None
    status: str
    top1_acc: float | None = None
    latency_ms: float | None = None
    parameter_count_million: float | None = None
    normalized_config_notes: list[str] = Field(default_factory=list)


class ModelCompareSummary(BaseModel):
    """Cross-model compare result summary."""

    mode: Literal["model_compare"] = "model_compare"
    shared_baseline_config: dict[str, Any]
    candidate_results: list[ModelCompareCandidateResult] = Field(default_factory=list)
    ai_summary: str | None = None
    ai_summary_error: str | None = None


class ModelCompareTaskResponse(BaseModel):
    """Background cross-model compare task snapshot."""

    task_id: str
    title: str | None = None
    status: str
    dataset: str | None = None
    candidate_models: list[str] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    current_model_name: str | None = None
    current_model_index: int = 0
    total_models: int = 0
    current_run_id: str | None = None
    current_experiment_id: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    activity_message: str | None = None
    dataset_summary: str | None = None
    training_image_size: int | None = None
    ai_model_name: str | None = None
    logs: list[str]
    summary: ModelCompareSummary | None = None
    error: str | None = None
    stop_requested: bool = False
    stop_reason: str | None = None


class TaskHistoryItemResponse(BaseModel):
    """Compact task item for workspace and history listings."""

    task_id: str
    task_type: Literal["auto_train", "model_compare"]
    title: str
    status: str
    summary: str | None = None
    dataset: str | None = None
    model_name: str | None = None
    candidate_models: list[str] = Field(default_factory=list)
    policy_preset: str | None = None
    run_id: str | None = None
    source_task_type: Literal["model_compare"] | None = None
    source_task_id: str | None = None
    source_task_title: str | None = None
    source_model_name: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class TaskTitleUpdateRequest(BaseModel):
    """Rename one task from the workspace header."""

    title: str
