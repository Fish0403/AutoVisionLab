"""Model APIs for parameter spaces."""

from fastapi import APIRouter, HTTPException

from app.api.responses import build_success_response
from app.schemas.api import ApiResponse
from app.schemas.parameter_space import EditableParameterSpace
from app.services.parameter_space import get_parameter_space


router = APIRouter()


@router.get("/{model_name}/parameter-space", response_model=ApiResponse[EditableParameterSpace])
def read_parameter_space(model_name: str) -> ApiResponse[EditableParameterSpace]:
    """Return the editable parameter space for a supported model."""
    parameter_space = get_parameter_space(model_name)
    if parameter_space is None:
        raise HTTPException(status_code=404, detail="Model is not supported")
    return build_success_response(parameter_space, message="Parameter space loaded.")
