"""Run-facing API schemas."""

from typing import Any

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
    experiments: list[ExperimentSummary]


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
    rounds: int = Field(ge=1, le=20)


class AutoTrainTaskResponse(BaseModel):
    """Background auto-train task snapshot."""

    task_id: str
    status: str
    run_id: str | None = None
    current_round: int = 0
    total_rounds: int
    current_experiment_id: str | None = None
    logs: list[str]
    summary: dict[str, Any] | None = None
    error: str | None = None
    stop_requested: bool = False
