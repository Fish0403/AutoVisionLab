"""Experiment APIs for the MVP skeleton."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db_session
from app.schemas.ai import ResultSchema
from app.schemas.experiment import ExperimentCreateRequest, ExperimentDecisionRequest, ExperimentDetailResponse
from app.services.persistence import (
    create_experiment,
    discard_experiment,
    get_experiment_config,
    get_experiment_detail,
    save_experiment_result,
    update_experiment_decision,
    update_experiment_status,
)
from app.services.training_runner import start_experiment_training, stop_experiment_training


router = APIRouter()


@router.post("", response_model=ExperimentDetailResponse, status_code=201)
def create_experiment_endpoint(
    request: ExperimentCreateRequest,
    db: Session = Depends(get_db_session),
) -> ExperimentDetailResponse:
    """Persist one experiment."""
    try:
        experiment = create_experiment(db, request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if experiment is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return experiment


@router.get("/{experiment_id}", response_model=ExperimentDetailResponse)
def get_experiment(experiment_id: str, db: Session = Depends(get_db_session)) -> ExperimentDetailResponse:
    """Get experiment details including params, result, proposal, and reflection."""
    experiment = get_experiment_detail(db, experiment_id)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@router.post("/{experiment_id}/result", response_model=ExperimentDetailResponse)
def save_result_endpoint(
    experiment_id: str,
    request: ResultSchema,
    db: Session = Depends(get_db_session),
) -> ExperimentDetailResponse:
    """Create or replace the structured result for one experiment."""
    experiment = save_experiment_result(db, experiment_id=experiment_id, result=request)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@router.post("/{experiment_id}/decision", response_model=ExperimentDetailResponse)
def save_decision_endpoint(
    experiment_id: str,
    request: ExperimentDecisionRequest,
    db: Session = Depends(get_db_session),
) -> ExperimentDetailResponse:
    """Create or replace the research decision for one experiment."""
    experiment = update_experiment_decision(db, experiment_id=experiment_id, request=request)
    if experiment is None:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiment


@router.post("/{experiment_id}/train", response_model=ExperimentDetailResponse)
def train_experiment_endpoint(
    experiment_id: str,
    db: Session = Depends(get_db_session),
) -> ExperimentDetailResponse:
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
    return refreshed_experiment


@router.post("/{experiment_id}/stop", response_model=ExperimentDetailResponse)
def stop_experiment_endpoint(
    experiment_id: str,
    db: Session = Depends(get_db_session),
) -> ExperimentDetailResponse:
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
    return discarded_experiment
