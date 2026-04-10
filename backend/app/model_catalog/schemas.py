"""Schemas for externally declared model manifests."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.parameter_space import EditableParameterSpace, ModelRecipe, SearchPolicy, TrainHyp
from app.schemas.ranking_policy import RankingPolicy


class TorchvisionClassifierBuilderSpec(BaseModel):
    """Builder spec for one generic torchvision classification model."""

    type: Literal["torchvision_classifier"]
    torchvision_name: str
    backbone_component_name: str
    backbone_extractor: Literal["feature_sequence", "feature_sequence_with_avgpool", "resnet_stages"]
    native_head_attr: str
    feature_dim: int = Field(gt=0)
    default_dropout: float = Field(default=0.0, ge=0, le=1)
    uses_dropout_arg: bool = False
    supported_neck_names: list[str] = Field(default_factory=lambda: ["avg_pool", "gem_pool"])
    supported_head_names: list[str] = Field(
        default_factory=lambda: ["native_classifier", "linear", "dropout_linear"]
    )
    default_neck_name: str = "avg_pool"
    default_head_name: str = "native_classifier"


class GoogLeNetClassifierBuilderSpec(BaseModel):
    """Builder spec for the special-case GoogLeNet path."""

    type: Literal["googlenet_classifier"]


ModelBuilderSpec = TorchvisionClassifierBuilderSpec | GoogLeNetClassifierBuilderSpec
LossAdapterType = Literal["default", "googlenet_aux"]


class ModelManifest(BaseModel):
    """One externally declared model entry loaded into the runtime catalog."""

    model_name: str
    label: str
    task_type: Literal["classification"]
    model_family: str
    builder: ModelBuilderSpec
    loss_adapter: LossAdapterType = "default"
    supports_compare: bool = True
    supports_search: bool = True
    is_default: bool = False
    display_order: int = 100
    default_model_recipe: ModelRecipe
    parameter_space: EditableParameterSpace

    @model_validator(mode="after")
    def validate_consistency(self) -> "ModelManifest":
        """Validate cross-field consistency for one loaded manifest."""
        if self.default_model_recipe.task_type != self.task_type:
            raise ValueError("default_model_recipe.task_type must match manifest task_type")
        if self.default_model_recipe.base_model != self.model_name:
            raise ValueError("default_model_recipe.base_model must match manifest model_name")
        if self.default_model_recipe.model_family != self.model_family:
            raise ValueError("default_model_recipe.model_family must match manifest model_family")
        if self.parameter_space.model_name != self.model_name:
            raise ValueError("parameter_space.model_name must match manifest model_name")
        if self.builder.type == "torchvision_classifier":
            if self.builder.default_neck_name not in set(self.builder.supported_neck_names):
                raise ValueError("builder.default_neck_name must exist in builder.supported_neck_names")
            if self.builder.default_head_name not in set(self.builder.supported_head_names):
                raise ValueError("builder.default_head_name must exist in builder.supported_head_names")
        return self


class ModelCatalogEntry(BaseModel):
    """Runtime-facing catalog entry."""

    model_name: str
    label: str
    task_type: Literal["classification"]
    model_family: str
    supports_compare: bool = True
    supports_search: bool = True
    is_default: bool = False
    display_order: int = 100
    builder: ModelBuilderSpec
    loss_adapter: LossAdapterType = "default"
    default_model_recipe: ModelRecipe
    parameter_space: EditableParameterSpace

    @classmethod
    def from_manifest(cls, manifest: ModelManifest) -> "ModelCatalogEntry":
        """Build one runtime entry from one validated manifest."""
        return cls.model_validate(manifest.model_dump(mode="python"))


class ModelSummary(BaseModel):
    """Frontend-facing summary for one registered model."""

    model_name: str
    label: str
    task_type: Literal["classification"]
    model_family: str
    supports_compare: bool = True
    supports_search: bool = True
    is_default: bool = False
    display_order: int = 100

    @classmethod
    def from_entry(cls, entry: ModelCatalogEntry) -> "ModelSummary":
        """Project one full catalog entry into a compact API summary."""
        return cls.model_validate(
            {
                "model_name": entry.model_name,
                "label": entry.label,
                "task_type": entry.task_type,
                "model_family": entry.model_family,
                "supports_compare": entry.supports_compare,
                "supports_search": entry.supports_search,
                "is_default": entry.is_default,
                "display_order": entry.display_order,
            }
        )


class ModelDefaults(BaseModel):
    """Default workspace-facing config template for one registered model."""

    summary: ModelSummary
    parameter_space: EditableParameterSpace
    default_model_recipe: ModelRecipe
    default_train_hyp: TrainHyp
    default_search_policy: SearchPolicy
    default_ranking_policy: RankingPolicy
