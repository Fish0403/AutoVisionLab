"""Run-facing API schemas."""

from typing import Any

from pydantic import BaseModel

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
    baseline_experiment_id: str | None = None
    best_quality_experiment_id: str | None = None
    best_experiment_id: str | None = None
    best_efficiency_experiment_id: str | None = None
    best_tradeoff_experiment_id: str | None = None
    frontier_experiment_id: str | None = None
    experiments: list[ExperimentSummary]


class RunSummaryResponse(BaseModel):
    """Compact run research summary."""

    run_id: str
    baseline_experiment_id: str | None = None
    best_quality_experiment_id: str | None = None
    best_experiment_id: str | None = None
    best_efficiency_experiment_id: str | None = None
    best_tradeoff_experiment_id: str | None = None
    frontier_experiment_id: str | None = None
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

    run_id: str | None = None
    run_name: str
    dataset: str
    model_name: str
    config: ExperimentConfig
    parameter_space: EditableParameterSpace


class AutoTrainTaskResponse(BaseModel):
    """Background auto-train task snapshot."""

    task_id: str
    status: str
    run_id: str | None = None
    current_round: int = 0
    elapsed_seconds: float = 0.0
    current_experiment_id: str | None = None
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
