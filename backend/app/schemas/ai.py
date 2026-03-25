"""Structured AI proposal and reflection schemas."""

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.common import ArtifactPaths, MetricsSnapshot, ReflectionOutcome, ResourceUsage, ResultStatus, RiskLevel
from app.schemas.parameter_space import ExperimentParams


class ProposalChanges(BaseModel):
    """Partial parameter update proposed by AI."""

    optimizer: str | None = None
    learning_rate: float | None = Field(default=None, gt=0)
    batch_size: int | None = Field(default=None, gt=0)
    image_size: int | None = Field(default=None, gt=0)
    epochs: int | None = Field(default=None, gt=0)
    weight_decay: float | None = Field(default=None, ge=0)
    scheduler: str | None = None
    augmentation_policy: str | None = None
    mixup_alpha: float | None = Field(default=None, ge=0)
    cutmix_alpha: float | None = Field(default=None, ge=0)
    random_erasing_prob: float | None = Field(default=None, ge=0, le=1)
    loss_name: str | None = None
    focal_gamma: float | None = Field(default=None, gt=0)
    label_smoothing: float | None = Field(default=None, ge=0, le=0.2)
    aux_logits: bool | None = None


class ProposalSchema(BaseModel):
    """AI-generated structured proposal."""

    task_type: Literal["classification"] = "classification"
    model_name: str
    based_on_experiment_ids: list[str]
    hypothesis: str
    changes: ProposalChanges
    reason: str
    risk: RiskLevel


class ResultSchema(BaseModel):
    """Structured training result."""

    status: ResultStatus
    metrics: MetricsSnapshot
    resource: ResourceUsage
    params: ExperimentParams
    artifacts: ArtifactPaths


class ReflectionSchema(BaseModel):
    """AI-generated structured reflection."""

    outcome: ReflectionOutcome
    analysis: str
    confidence: float = Field(ge=0, le=1)
    next_action: str
    recommended_changes: ProposalChanges
