"""Editable parameter space schemas."""

from typing import Literal

from pydantic import BaseModel, Field


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


class ExperimentParams(BaseModel):
    """Structured training parameters allowed in the MVP."""

    optimizer: str
    learning_rate: float = Field(gt=0)
    batch_size: int = Field(gt=0)
    image_size: int = Field(gt=0)
    epochs: int = Field(gt=0)
    weight_decay: float = Field(ge=0)
    scheduler: str
    augmentation_level: str
    label_smoothing: float = Field(ge=0, le=0.2)
    aux_logits: bool | None = None


class ExperimentConfig(BaseModel):
    """Final config consumed by the trainer."""

    task_type: Literal["classification"]
    dataset: str
    model_family: Literal["mobilenet", "googlenet"]
    model_name: Literal["mobilenet_v2", "googlenet"]
    parameter_space_version: str
    params: ExperimentParams

