"""Dataset discovery APIs."""

from fastapi import APIRouter

from app.api.responses import build_success_response
from app.schemas.api import ApiResponse
from app.schemas.dataset import LocalDatasetSummary
from app.services.dataset_service import list_local_datasets


router = APIRouter()


@router.get("", response_model=ApiResponse[list[LocalDatasetSummary]])
def get_datasets() -> ApiResponse[list[LocalDatasetSummary]]:
    """List locally discovered datasets and readiness status."""
    return build_success_response(list_local_datasets(), message="Datasets loaded.")
