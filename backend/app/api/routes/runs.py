"""Run APIs for the MVP skeleton."""

from fastapi import APIRouter, Depends, HTTPException
import requests
from sqlalchemy.orm import Session

from app.api.responses import build_success_response
from app.db.session import get_db_session
from app.schemas.api import ApiResponse
from app.schemas.ai import ProposalSchema
from app.schemas.run import (
    AutoTrainStartRequest,
    AutoTrainTaskResponse,
    ModelCompareStartRequest,
    ModelCompareTaskResponse,
    RunCreateRequest,
    RunDetailResponse,
    RunListItem,
    RunMetricsResponse,
    RunSummaryResponse,
    TaskHistoryItemResponse,
    TaskTitleUpdateRequest,
)
from app.services.auto_train_service import (
    get_active_auto_train_task,
    get_auto_train_task,
    list_auto_train_tasks,
    start_auto_train_task,
    stop_auto_train_task,
    update_auto_train_task_title,
)
from app.services.model_compare_service import (
    get_active_model_compare_task,
    get_model_compare_task,
    list_model_compare_tasks,
    start_model_compare_task,
    stop_model_compare_task,
    update_model_compare_task_title,
)
from app.services.persistence import clear_all_records, clear_run_records, create_run, get_run_detail, get_run_metrics, get_run_summary, list_runs
from app.services.proposal_service import generate_aihubmix_proposal, test_aihubmix_connection


router = APIRouter()


@router.get("", response_model=ApiResponse[list[RunListItem]])
def get_runs(db: Session = Depends(get_db_session)) -> ApiResponse[list[RunListItem]]:
    """List all runs."""
    return build_success_response(list_runs(db), message="Runs loaded.")


@router.get("/tasks", response_model=ApiResponse[list[TaskHistoryItemResponse]])
def get_task_history() -> ApiResponse[list[TaskHistoryItemResponse]]:
    """List auto-train and model-compare tasks in one unified history feed."""
    task_items = list_auto_train_tasks() + list_model_compare_tasks()
    task_items.sort(key=lambda item: item.updated_at or "", reverse=True)
    return build_success_response(task_items, message="Task history loaded.")


@router.post("/tasks/{task_type}/{task_id}/title", response_model=ApiResponse[dict[str, str]])
def update_task_title(task_type: str, task_id: str, request: TaskTitleUpdateRequest) -> ApiResponse[dict[str, str]]:
    """Update one task title for workspace and history views."""
    if task_type == "auto_train":
        task = update_auto_train_task_title(task_id, request.title)
    elif task_type == "model_compare":
        task = update_model_compare_task_title(task_id, request.title)
    else:
        raise HTTPException(status_code=404, detail="Task type not found")
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return build_success_response({"task_id": task.task_id, "title": task.title or request.title}, message="Task title updated.")


@router.post("/model-compare", response_model=ApiResponse[ModelCompareTaskResponse], status_code=202)
def start_model_compare_endpoint(request: ModelCompareStartRequest) -> ApiResponse[ModelCompareTaskResponse]:
    """Start one background cross-model compare task."""
    try:
        task = start_model_compare_task(request)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return build_success_response(task, message="Model compare task started.", code=task.status)


@router.get("/model-compare/active", response_model=ApiResponse[ModelCompareTaskResponse])
def get_active_model_compare_endpoint() -> ApiResponse[ModelCompareTaskResponse]:
    """Get the currently active cross-model compare task when one exists."""
    task = get_active_model_compare_task()
    if task is None:
        raise HTTPException(status_code=404, detail="No active model compare task")
    return build_success_response(task, message="Active model compare task loaded.", code=task.status)


@router.get("/model-compare/{task_id}", response_model=ApiResponse[ModelCompareTaskResponse])
def get_model_compare_endpoint(task_id: str) -> ApiResponse[ModelCompareTaskResponse]:
    """Get one background cross-model compare task snapshot."""
    task = get_model_compare_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Model compare task not found")
    return build_success_response(task, message="Model compare task loaded.", code=task.status)


@router.post("/model-compare/{task_id}/stop", response_model=ApiResponse[ModelCompareTaskResponse])
def stop_model_compare_endpoint(task_id: str) -> ApiResponse[ModelCompareTaskResponse]:
    """Stop one background model-compare task."""
    task = stop_model_compare_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Model compare task not found")
    return build_success_response(task, message="Model compare stop requested.", code=task.status)


@router.post("", response_model=ApiResponse[RunDetailResponse], status_code=201)
def create_run_endpoint(request: RunCreateRequest, db: Session = Depends(get_db_session)) -> ApiResponse[RunDetailResponse]:
    """Persist one run."""
    try:
        run = create_run(db, request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return build_success_response(run, message="Run created.", code="created")


@router.get("/{run_id}", response_model=ApiResponse[RunDetailResponse])
def get_run(run_id: str, db: Session = Depends(get_db_session)) -> ApiResponse[RunDetailResponse]:
    """Get one run by identifier."""
    run = get_run_detail(db, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return build_success_response(run, message="Run detail loaded.")


@router.get("/{run_id}/metrics", response_model=ApiResponse[RunMetricsResponse])
def get_metrics(run_id: str, metric_name: str = "top1_acc", db: Session = Depends(get_db_session)) -> ApiResponse[RunMetricsResponse]:
    """Return trend data for one run."""
    metrics = get_run_metrics(db=db, run_id=run_id, metric_name=metric_name)
    if metrics is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return build_success_response(metrics, message="Run metrics loaded.")


@router.get("/{run_id}/summary", response_model=ApiResponse[RunSummaryResponse])
def get_run_summary_endpoint(run_id: str, db: Session = Depends(get_db_session)) -> ApiResponse[RunSummaryResponse]:
    """Return run-level anchors and decision counts."""
    summary = get_run_summary(db, run_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return build_success_response(summary, message="Run summary loaded.")


@router.post("/{run_id}/proposal", response_model=ApiResponse[ProposalSchema])
def generate_proposal_endpoint(run_id: str, db: Session = Depends(get_db_session)) -> ApiResponse[ProposalSchema]:
    """Generate one AIHubMix proposal for the given run."""
    try:
        proposal = generate_aihubmix_proposal(db, run_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except requests.RequestException as error:
        response_text = error.response.text if error.response is not None else str(error)
        raise HTTPException(status_code=502, detail=f"AIHubMix request failed: {response_text}") from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return build_success_response(proposal, message="Proposal generated.")


@router.post("/proposal/test", response_model=ApiResponse[dict[str, str]])
def test_proposal_provider() -> ApiResponse[dict[str, str]]:
    """Run a minimal AIHubMix connectivity test."""
    try:
        result = test_aihubmix_connection()
    except requests.RequestException as error:
        response_text = error.response.text if error.response is not None else str(error)
        raise HTTPException(status_code=502, detail=f"AIHubMix request failed: {response_text}") from error
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return build_success_response(result, message="Proposal provider connectivity verified.")


@router.post("/reset", response_model=ApiResponse[dict[str, int]])
def reset_all_runs(db: Session = Depends(get_db_session)) -> ApiResponse[dict[str, int]]:
    """Delete all persisted runs, experiments, and results."""
    return build_success_response(clear_all_records(db), message="All runs cleared.")


@router.post("/{run_id}/reset", response_model=ApiResponse[dict[str, int]])
def reset_single_run(run_id: str, db: Session = Depends(get_db_session)) -> ApiResponse[dict[str, int]]:
    """Delete one run and all of its persisted experiments/results."""
    deleted_counts = clear_run_records(db, run_id)
    if deleted_counts is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return build_success_response(deleted_counts, message="Run cleared.")


@router.post("/auto-train", response_model=ApiResponse[AutoTrainTaskResponse], status_code=202)
def start_auto_train_endpoint(request: AutoTrainStartRequest) -> ApiResponse[AutoTrainTaskResponse]:
    """Start one background auto-train task."""
    try:
        task = start_auto_train_task(request)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return build_success_response(task, message="Auto-train task started.", code=task.status)


@router.get("/auto-train/active", response_model=ApiResponse[AutoTrainTaskResponse])
def get_active_auto_train_endpoint() -> ApiResponse[AutoTrainTaskResponse]:
    """Get the currently active auto-train task when one exists."""
    task = get_active_auto_train_task()
    if task is None:
        raise HTTPException(status_code=404, detail="No active auto train task")
    return build_success_response(task, message="Active auto-train task loaded.", code=task.status)


@router.get("/auto-train/{task_id}", response_model=ApiResponse[AutoTrainTaskResponse])
def get_auto_train_endpoint(task_id: str) -> ApiResponse[AutoTrainTaskResponse]:
    """Get one background auto-train task snapshot."""
    task = get_auto_train_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Auto train task not found")
    return build_success_response(task, message="Auto-train task loaded.", code=task.status)


@router.post("/auto-train/{task_id}/stop", response_model=ApiResponse[AutoTrainTaskResponse])
def stop_auto_train_endpoint(task_id: str) -> ApiResponse[AutoTrainTaskResponse]:
    """Stop one background auto-train task."""
    task = stop_auto_train_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Auto train task not found")
    return build_success_response(task, message="Auto-train stop requested.", code=task.status)
