"""Model APIs for parameter spaces."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.responses import build_success_response
from app.model_catalog.registry import list_model_summaries
from app.model_catalog.schemas import ModelDefaults, ModelSummary
from app.schemas.api import ApiResponse
from app.schemas.model_manifest_draft import (
    ModelManifestCommitRequest,
    ModelManifestCommitResponse,
    ModelManifestDraftRequest,
    ModelManifestDraftResponse,
)
from app.schemas.parameter_space import EditableParameterSpace
from app.services.model_defaults_service import get_model_defaults
from app.services.model_manifest_draft_service import commit_model_manifest, draft_model_manifest
from app.services.parameter_space import get_parameter_space
from app.llm.aihubmix_client import AIHubMixRequestError


router = APIRouter()


@router.get("", response_model=ApiResponse[list[ModelSummary]])
def list_models() -> ApiResponse[list[ModelSummary]]:
    """Return all registered models available to the workspace."""
    return build_success_response(list_model_summaries(), message="Models loaded.")


@router.post("/draft", response_model=ApiResponse[ModelManifestDraftResponse])
def draft_model_manifest_endpoint(request: ModelManifestDraftRequest) -> ApiResponse[ModelManifestDraftResponse]:
    """Generate one model manifest draft without persisting it."""
    try:
        draft_response = draft_model_manifest(request)
    except (AIHubMixRequestError, ValueError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return build_success_response(draft_response, message="Model manifest draft generated.")


@router.post("/draft/commit", response_model=ApiResponse[ModelManifestCommitResponse])
def commit_model_manifest_endpoint(request: ModelManifestCommitRequest) -> ApiResponse[ModelManifestCommitResponse]:
    """Persist one approved model manifest draft."""
    try:
        commit_response = commit_model_manifest(request)
    except FileExistsError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return build_success_response(commit_response, message="Model manifest committed.", code="created")


@router.get("/{model_name}/defaults", response_model=ApiResponse[ModelDefaults])
def read_model_defaults(model_name: str) -> ApiResponse[ModelDefaults]:
    """Return the default config template for one supported model."""
    model_defaults = get_model_defaults(model_name)
    if model_defaults is None:
        raise HTTPException(status_code=404, detail="Model is not supported")
    return build_success_response(model_defaults, message="Model defaults loaded.")


@router.get("/{model_name}/parameter-space", response_model=ApiResponse[EditableParameterSpace])
def read_parameter_space(model_name: str) -> ApiResponse[EditableParameterSpace]:
    """Return the editable parameter space for a supported model."""
    parameter_space = get_parameter_space(model_name)
    if parameter_space is None:
        raise HTTPException(status_code=404, detail="Model is not supported")
    return build_success_response(parameter_space, message="Parameter space loaded.")
