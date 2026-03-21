"""Run APIs for the MVP skeleton."""

from fastapi import APIRouter, Depends, HTTPException
import requests
from sqlalchemy.orm import Session

from app.db.session import get_db_session
from app.schemas.ai import ProposalSchema
from app.schemas.run import AutoTrainStartRequest, AutoTrainTaskResponse, RunCreateRequest, RunDetailResponse, RunListItem, RunMetricsResponse
from app.services.auto_train_service import get_auto_train_task, start_auto_train_task, stop_auto_train_task
from app.services.persistence import clear_all_records, clear_run_records, create_run, get_run_detail, get_run_metrics, list_runs
from app.services.proposal_service import generate_aihubmix_proposal, test_aihubmix_connection


router = APIRouter()


@router.get("", response_model=list[RunListItem])
def get_runs(db: Session = Depends(get_db_session)) -> list[RunListItem]:
    """List all runs."""
    return list_runs(db)


@router.post("", response_model=RunDetailResponse, status_code=201)
def create_run_endpoint(request: RunCreateRequest, db: Session = Depends(get_db_session)) -> RunDetailResponse:
    """Persist one run."""
    try:
        return create_run(db, request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/{run_id}", response_model=RunDetailResponse)
def get_run(run_id: str, db: Session = Depends(get_db_session)) -> RunDetailResponse:
    """Get one run by identifier."""
    run = get_run_detail(db, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/{run_id}/metrics", response_model=RunMetricsResponse)
def get_metrics(run_id: str, metric_name: str = "top1_acc", db: Session = Depends(get_db_session)) -> RunMetricsResponse:
    """Return trend data for one run."""
    metrics = get_run_metrics(db=db, run_id=run_id, metric_name=metric_name)
    if metrics is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return metrics


@router.post("/{run_id}/proposal", response_model=ProposalSchema)
def generate_proposal_endpoint(run_id: str, db: Session = Depends(get_db_session)) -> ProposalSchema:
    """Generate one AIHubMix proposal for the given run."""
    try:
        return generate_aihubmix_proposal(db, run_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except requests.RequestException as error:
        response_text = error.response.text if error.response is not None else str(error)
        raise HTTPException(status_code=502, detail=f"AIHubMix request failed: {response_text}") from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@router.post("/proposal/test")
def test_proposal_provider() -> dict:
    """Run a minimal AIHubMix connectivity test."""
    try:
        return test_aihubmix_connection()
    except requests.RequestException as error:
        response_text = error.response.text if error.response is not None else str(error)
        raise HTTPException(status_code=502, detail=f"AIHubMix request failed: {response_text}") from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error


@router.post("/reset")
def reset_all_runs(db: Session = Depends(get_db_session)) -> dict[str, int]:
    """Delete all persisted runs, experiments, and results."""
    return clear_all_records(db)


@router.post("/{run_id}/reset")
def reset_single_run(run_id: str, db: Session = Depends(get_db_session)) -> dict[str, int]:
    """Delete one run and all of its persisted experiments/results."""
    deleted_counts = clear_run_records(db, run_id)
    if deleted_counts is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return deleted_counts


@router.post("/auto-train", response_model=AutoTrainTaskResponse, status_code=202)
def start_auto_train_endpoint(request: AutoTrainStartRequest) -> AutoTrainTaskResponse:
    """Start one background auto-train task."""
    try:
        return start_auto_train_task(request)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get("/auto-train/{task_id}", response_model=AutoTrainTaskResponse)
def get_auto_train_endpoint(task_id: str) -> AutoTrainTaskResponse:
    """Get one background auto-train task snapshot."""
    task = get_auto_train_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Auto train task not found")
    return task


@router.post("/auto-train/{task_id}/stop", response_model=AutoTrainTaskResponse)
def stop_auto_train_endpoint(task_id: str) -> AutoTrainTaskResponse:
    """Stop one background auto-train task."""
    task = stop_auto_train_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Auto train task not found")
    return task
