"""Schemas for AI-assisted model manifest drafting."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.model_catalog.schemas import ModelBuilderSpec


class ModelManifestDraftRequest(BaseModel):
    """Request payload for drafting one model manifest."""

    query: str = Field(min_length=1, max_length=120)


class ModelManifestValidationResult(BaseModel):
    """Validation summary for one drafted manifest."""

    is_valid: bool
    yaml_parse_ok: bool = False
    schema_ok: bool = False
    dry_run_build_ok: bool = False
    dry_run_forward_ok: bool = False
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ModelManifestDraftResponse(BaseModel):
    """Draft preview returned before any manifest is committed."""

    query: str
    resolved_model_name: str
    ai_preview_text: str
    yaml_text: str
    target_path: str
    provider_warnings: list[str] = Field(default_factory=list)
    validation: ModelManifestValidationResult


DraftRecipeProfile = Literal["torchvision_classifier_v1", "googlenet_classifier_v1"]
DraftParameterSpaceProfile = Literal[
    "classification_standard_v1",
    "classification_memory_safe_v1",
    "googlenet_aux_v1",
]


class MinimalModelManifestDraftSpec(BaseModel):
    """Minimal AI-generated draft that the backend expands into one full manifest."""

    model_name: str
    label: str
    model_family: str
    builder: ModelBuilderSpec
    recipe_profile: DraftRecipeProfile
    parameter_space_profile: DraftParameterSpaceProfile
    supports_compare: bool = True
    supports_search: bool = True
    is_default: bool = False
    display_order: int = 120


class ModelManifestCommitRequest(BaseModel):
    """Commit request for one approved manifest draft."""

    yaml_text: str = Field(min_length=1)
    expected_model_name: str | None = None


class ModelManifestCommitResponse(BaseModel):
    """Response returned after one manifest is committed."""

    model_name: str
    manifest_path: str
    reloaded_model_count: int = Field(ge=0)
