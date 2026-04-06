"""Structured AI proposal and reflection schemas."""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.common import ArtifactPaths, MetricsSnapshot, ReflectionOutcome, ResourceUsage, ResultStatus
from app.schemas.parameter_space import (
    ExperimentParams,
    build_model_recipe_change_payload,
    build_train_hyp_change_payload,
)


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
    width_multiple: float | None = Field(default=None, gt=0)
    pooling_type: str | None = None
    classifier_dropout: float | None = Field(default=None, ge=0, le=1)
    neck_name: str | None = None
    head_name: str | None = None


class ProposalSchema(BaseModel):
    """AI-generated structured proposal."""

    task_type: Literal["classification"] = "classification"
    model_name: str
    based_on_experiment_ids: list[str]
    hypothesis: str
    changes: ProposalChanges
    train_hyp_changes: dict[str, Any] | None = None
    recipe_changes: dict[str, Any] | None = None
    reason: str

    @model_validator(mode="after")
    def populate_recipe_change_views(self) -> "ProposalSchema":
        """Backfill recipe-oriented change views from the legacy flat change map."""
        effective_changes = self.changes.model_dump(exclude_none=True)
        if self.train_hyp_changes is None:
            derived_train_hyp_changes = build_train_hyp_change_payload(effective_changes)
            self.train_hyp_changes = derived_train_hyp_changes or None
        if self.recipe_changes is None:
            derived_recipe_changes = build_model_recipe_change_payload(effective_changes)
            self.recipe_changes = derived_recipe_changes or None
        return self


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
