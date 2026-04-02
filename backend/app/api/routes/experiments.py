"""Experiment APIs for the MVP skeleton."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.responses import build_success_response
from app.db.session import get_db_session
from app.schemas.ai import ResultSchema
from app.schemas.api import ApiResponse
from app.schemas.experiment import (
    ExperimentCreateRequest,
    ExperimentDetailResponse,
    ExperimentSuggestionTaskResponse,
)
from app.services.persistence import (
    create_experiment,
    discard_experiment,
    get_experiment_config,
    get_experiment_detail,
    save_experiment_result,
)
from app.services.suggestion_service import get_experiment_suggestion_task, start_experiment_suggestion_task
from app.services.training_runner import start_experiment_training, stop_experiment_training


router = APIRouter()


@router.post("", response_model=ApiResponse[ExperimentDetailResponse], status_code=201)
def create_experiment_endpoint(
    request: ExperimentCreateRequest,
    db: Session = Depends(get_db_session),
) -> ApiResponse[ExperimentDetailResponse]:
    """Persist one experiment."""
    try:
        experiment = create_experiment(db, request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if experiment is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return build_success_response(experiment, message="Experiment created.", code="created")


@router.get("/{experiment_id}", response_model=ApiResponse[ExperimentDetailResponse])
def get_experiment(experiment_id: str, db: Session = Depends(get_db_session)) -> ApiResponse[ExperimentDetailResponse]:
    """Get experiment details including params, result, proposal, and reflection."""
    experiment = get_experiment_detail(db, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return build_success_response(experiment, message="Experiment detail loaded.")


@router.post("/{experiment_id}/result", response_model=ApiResponse[ExperimentDetailResponse])
def save_result_endpoint(
    experiment_id: str,
    request: ResultSchema,
    db: Session = Depends(get_db_session),
) -> ApiResponse[ExperimentDetailResponse]:
    """Create or replace the structured result for one experiment."""
    experiment = save_experiment_result(db, experiment_id=experiment_id, result=request)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return build_success_response(experiment, message="Experiment result saved.", code="updated")


@router.post("/{experiment_id}/train", response_model=ApiResponse[ExperimentDetailResponse])
def train_experiment_endpoint(
    experiment_id: str,
    db: Session = Depends(get_db_session),
) -> ApiResponse[ExperimentDetailResponse]:
    """Start one experiment in the background for the demo backend."""
    config = get_experiment_config(db, experiment_id)
    if config is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    try:
        started_experiment = start_experiment_training(experiment_id)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    if started_experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    refreshed_experiment = get_experiment_detail(db, experiment_id)
    if refreshed_experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return build_success_response(
        refreshed_experiment,
        message="Experiment training started.",
        code=refreshed_experiment.status,
    )


@router.post("/{experiment_id}/stop", response_model=ApiResponse[ExperimentDetailResponse])
def stop_experiment_endpoint(
    experiment_id: str,
    db: Session = Depends(get_db_session),
) -> ApiResponse[ExperimentDetailResponse]:
    """Stop the current running experiment and discard it."""
    try:
        stopped_experiment = stop_experiment_training(experiment_id)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    if stopped_experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    discarded_experiment = discard_experiment(db, experiment_id)
    if discarded_experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return build_success_response(
        discarded_experiment,
        message="Experiment stop requested.",
        code=discarded_experiment.status,
    )


@router.post("/{experiment_id}/suggestion", response_model=ApiResponse[ExperimentSuggestionTaskResponse], status_code=202)
def start_experiment_suggestion_endpoint(experiment_id: str) -> ApiResponse[ExperimentSuggestionTaskResponse]:
    """Start or reuse one background suggestion task for a completed experiment."""
    try:
        task = start_experiment_suggestion_task(experiment_id)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return build_success_response(task, message="Experiment suggestion task started.", code=task.status)


@router.get("/{experiment_id}/suggestion", response_model=ApiResponse[ExperimentSuggestionTaskResponse])
def get_experiment_suggestion_endpoint(experiment_id: str) -> ApiResponse[ExperimentSuggestionTaskResponse]:
    """Get the latest background suggestion task for one experiment."""
    task = get_experiment_suggestion_task(experiment_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Experiment suggestion task not found")
    return build_success_response(task, message="Experiment suggestion task loaded.", code=task.status)
