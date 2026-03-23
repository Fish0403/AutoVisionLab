"""Experiment-facing API schemas."""

from pydantic import BaseModel

from app.schemas.ai import ProposalSchema, ReflectionSchema, ResultSchema
from app.schemas.common import ExperimentDecision, ExperimentStatus
from app.schemas.parameter_space import EditableParameterSpace, ExperimentConfig


class ExperimentSummary(BaseModel):
    """Compact experiment row."""

    id: str
    run_id: str
    status: ExperimentStatus
    model_name: str
    decision: ExperimentDecision | None = None
    is_best_so_far: bool = False


class ExperimentCreateRequest(BaseModel):
    """Request payload for creating one experiment under a run."""

    run_id: str
    config: ExperimentConfig
    parameter_space: EditableParameterSpace
    proposal: ProposalSchema | None = None


class ExperimentDetailResponse(BaseModel):
    """Full experiment payload for the detail panel."""

    id: str
    run_id: str
    status: ExperimentStatus
    decision: ExperimentDecision | None = None
    decision_reason: str | None = None
    baseline_experiment_id: str | None = None
    is_best_so_far: bool = False
    config: ExperimentConfig
    parameter_space: EditableParameterSpace
    proposal: ProposalSchema | None = None
    result: ResultSchema | None = None
    reflection: ReflectionSchema | None = None


class ExperimentDecisionRequest(BaseModel):
    """Request payload for writing one research decision."""

    decision: ExperimentDecision
    decision_reason: str | None = None
