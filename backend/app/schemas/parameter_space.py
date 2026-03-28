"""Editable parameter space schemas."""

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.ranking_policy import RankingPolicy


DEFAULT_BASIC_HPARAM_SEARCH_FIELDS = [
    "optimizer",
    "learning_rate",
    "batch_size",
    "weight_decay",
    "scheduler",
    "label_smoothing",
]


class EnumParamDefinition(BaseModel):
    """Parameter with a fixed choice set."""

    type: Literal["enum"]
    choices: list[str | int | bool]


class NumberRangeParamDefinition(BaseModel):
    """Parameter with a numeric range."""

    type: Literal["number_range"]
    min: float
    max: float


class DiscreteValuesParamDefinition(BaseModel):
    """Parameter with discrete allowed numeric values."""

    type: Literal["discrete_values"]
    choices: list[int | float]


ParameterDefinition = EnumParamDefinition | NumberRangeParamDefinition | DiscreteValuesParamDefinition


class EditableParameterSpace(BaseModel):
    """White-listed editable parameter space for a model."""

    model_name: str
    version: str
    editable_params: dict[str, ParameterDefinition]


class LossParams(BaseModel):
    """Structured parameters for supported loss functions."""

    focal_gamma: float = Field(default=2.0, gt=0)


class AugmentationParams(BaseModel):
    """Structured parameters for supported augmentation strategies."""

    mixup_alpha: float = Field(default=0.0, ge=0)
    cutmix_alpha: float = Field(default=0.0, ge=0)
    random_erasing_prob: float = Field(default=0.0, ge=0, le=1)


class ExperimentParams(BaseModel):
    """Structured training parameters allowed in the MVP."""

    optimizer: str
    learning_rate: float = Field(gt=0)
    batch_size: int = Field(gt=0)
    image_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    weight_decay: float = Field(ge=0)
    scheduler: str
    augmentation_policy: Literal["none", "basic"] = "basic"
    augmentation_params: AugmentationParams = Field(default_factory=AugmentationParams)
    loss_name: Literal["cross_entropy", "cross_entropy_with_label_smoothing", "focal_loss"] = (
        "cross_entropy_with_label_smoothing"
    )
    loss_params: LossParams = Field(default_factory=LossParams)
    label_smoothing: float = Field(ge=0, le=0.2)
    aux_logits: bool | None = None


class SearchPolicy(BaseModel):
    """Controls what the AI is allowed to search automatically."""

    allow_basic_hparam_search: bool = True
    allowed_basic_hparam_fields: list[str] = Field(default_factory=lambda: list(DEFAULT_BASIC_HPARAM_SEARCH_FIELDS))
    allow_strategy_search: bool = False
    allow_loss_search: bool = False
    allow_augmentation_search: bool = False
    require_manual_approval_for_high_impact_changes: bool = True


class ExperimentConfig(BaseModel):
    """Final config consumed by the trainer."""

    task_type: Literal["classification"]
    dataset: str
    model_family: Literal["mobilenet", "googlenet", "resnet", "densenet"]
    model_name: Literal["mobilenet_v2", "googlenet", "resnet18", "resnet34", "densenet121"]
    parameter_space_version: str
    use_demo_mode: bool = False
    participates_in_ranking: bool = True
    search_policy: SearchPolicy = Field(default_factory=SearchPolicy)
    ranking_policy: RankingPolicy = Field(default_factory=RankingPolicy)
    params: ExperimentParams
