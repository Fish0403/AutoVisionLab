"""Persistence helpers for the SQLite demo backend."""

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.settings import get_settings
from app.models.experiment import ExperimentModel
from app.models.result import ResultModel
from app.models.run import RunModel
from app.services.run_logging import append_run_log
from app.services.run_policy import evaluate_promotion, get_default_run_policy
from app.schemas.ai import ProposalSchema, ReflectionSchema, ResultSchema
from app.schemas.common import PointMetric
from app.schemas.experiment import ExperimentCreateRequest, ExperimentDecisionRequest, ExperimentDetailResponse, ExperimentSummary
from app.schemas.parameter_space import EditableParameterSpace, ExperimentConfig
from app.schemas.run import RunCreateRequest, RunDetailResponse, RunListItem, RunMetricsResponse, RunSummaryResponse

def _refresh_run_summary(db: Session, run_id: str) -> RunModel | None:
    run = db.get(RunModel, run_id)
    if run is None:
        return None

    experiments = db.scalars(
        select(ExperimentModel).where(ExperimentModel.run_id == run_id).order_by(ExperimentModel.created_at.asc())
    ).all()
    if not experiments:
        run.baseline_experiment_id = None
        run.best_experiment_id = None
        run.frontier_experiment_id = None
        run.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(run)
        return run

    baseline_experiment = experiments[0]
    best_experiment: ExperimentModel | None = None
    frontier_experiment: ExperimentModel | None = None
    run_policy = get_default_run_policy()

    for experiment in experiments:
        experiment.is_best_so_far = False
        if experiment.status == "failed" and experiment.decision is None:
            experiment.decision = "crash"
            experiment.decision_reason = experiment.decision_reason or "Training failed before producing a valid result."
        if experiment.status == "discarded" and experiment.decision is None:
            experiment.decision = "discard"
            experiment.decision_reason = experiment.decision_reason or "Experiment was stopped or discarded before completion."
        if experiment.status == "success":
            if not experiment.experiment_config.get("participates_in_ranking", True):
                experiment.decision = "discard"
                experiment.decision_reason = "Completed successfully but does not participate in run ranking."
            else:
                should_promote, decision_reason = evaluate_promotion(
                    experiment,
                    best_experiment,
                    policy=run_policy,
                )
                if should_promote:
                    best_experiment = experiment
                    frontier_experiment = experiment
                    experiment.decision = "keep"
                    experiment.decision_reason = decision_reason
                else:
                    experiment.decision = "discard"
                    experiment.decision_reason = decision_reason
        experiment.updated_at = datetime.utcnow()

    if best_experiment is not None:
        best_experiment.is_best_so_far = True
    elif baseline_experiment.status == "success":
        best_experiment = baseline_experiment
        frontier_experiment = baseline_experiment
        baseline_experiment.is_best_so_far = True
        baseline_experiment.decision = "keep"
        baseline_experiment.decision_reason = "Promoted as the first successful experiment in the run."

    for experiment in experiments:
        if experiment.status == "success" and experiment.decision is None:
            if best_experiment is not None and experiment.id == best_experiment.id:
                experiment.decision = "keep"
                experiment.decision_reason = experiment.decision_reason or "Current best experiment under the run promotion policy."
            else:
                experiment.decision = "discard"
        experiment.updated_at = datetime.utcnow()

    run.baseline_experiment_id = baseline_experiment.id
    run.best_experiment_id = best_experiment.id if best_experiment is not None else None
    run.frontier_experiment_id = frontier_experiment.id if frontier_experiment is not None else None
    run.status = "active"
    run.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(run)
    return run


def _to_experiment_summary(experiment: ExperimentModel) -> ExperimentSummary:
    return ExperimentSummary(
        id=experiment.id,
        run_id=experiment.run_id,
        status=experiment.status,
        model_name=experiment.experiment_config["model_name"],
        decision=experiment.decision,
        is_best_so_far=bool(experiment.is_best_so_far),
    )


def _to_run_detail(db: Session, run: RunModel) -> RunDetailResponse:
    experiments = db.scalars(
        select(ExperimentModel).where(ExperimentModel.run_id == run.id).order_by(ExperimentModel.created_at.asc())
    ).all()
    return RunDetailResponse(
        id=run.id,
        name=run.name,
        dataset=run.dataset,
        model_name=run.model_name,
        status=run.status,
        notes=run.notes,
        baseline_experiment_id=run.baseline_experiment_id,
        best_experiment_id=run.best_experiment_id,
        frontier_experiment_id=run.frontier_experiment_id,
        experiments=[_to_experiment_summary(experiment) for experiment in experiments],
    )


def list_runs(db: Session) -> list[RunListItem]:
    """List all persisted runs."""
    runs = db.scalars(select(RunModel).order_by(RunModel.created_at.desc())).all()
    items: list[RunListItem] = []
    for run in runs:
        experiment_count = db.query(ExperimentModel).filter(ExperimentModel.run_id == run.id).count()
        items.append(
            RunListItem(
                id=run.id,
                name=run.name,
                dataset=run.dataset,
                model_name=run.model_name,
                status=run.status,
                experiment_count=experiment_count,
            )
        )
    return items


def create_run(db: Session, request: RunCreateRequest) -> RunDetailResponse:
    """Persist one run."""
    if request.model_name != request.base_config.model_name:
        raise ValueError("Run model_name must match base_config.model_name")
    if request.dataset != request.base_config.dataset:
        raise ValueError("Run dataset must match base_config.dataset")

    run = RunModel(
        id=f"run_{uuid4().hex[:8]}",
        name=request.name,
        task_type=request.base_config.task_type,
        dataset=request.dataset,
        model_name=request.model_name,
        status="active",
        notes=request.notes,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return _to_run_detail(db, run)


def get_run_detail(db: Session, run_id: str) -> RunDetailResponse | None:
    """Load one persisted run."""
    run = db.get(RunModel, run_id)
    if run is None:
        return None
    return _to_run_detail(db, run)


def create_experiment(db: Session, request: ExperimentCreateRequest) -> ExperimentDetailResponse | None:
    """Persist one experiment under an existing run."""
    run = db.get(RunModel, request.run_id)
    if run is None:
        return None
    if request.config.model_name != run.model_name:
        raise ValueError("Experiment model_name must match its parent run")
    if request.config.dataset != run.dataset:
        raise ValueError("Experiment dataset must match its parent run")
    if request.parameter_space.model_name != request.config.model_name:
        raise ValueError("Parameter space model_name must match experiment config model_name")

    experiment = ExperimentModel(
        id=f"exp_{uuid4().hex[:8]}",
        run_id=request.run_id,
        status="queued",
        decision=None,
        decision_reason=None,
        baseline_experiment_id=run.baseline_experiment_id,
        is_best_so_far=False,
        experiment_config=request.config.model_dump(),
        editable_parameter_space=request.parameter_space.model_dump(),
        proposal=request.proposal.model_dump() if request.proposal else None,
        result=None,
        reflection=None,
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    append_run_log(
        request.run_id,
        f"[{experiment.id}] experiment created"
        + (
            f" | based_on={','.join(request.proposal.based_on_experiment_ids)}"
            if request.proposal and request.proposal.based_on_experiment_ids
            else ""
        ),
    )
    _refresh_run_summary(db, request.run_id)
    db.refresh(experiment)
    return _to_experiment_detail(experiment)


def update_experiment_status(db: Session, experiment_id: str, status: str) -> ExperimentDetailResponse | None:
    """Update one experiment status."""
    experiment = db.get(ExperimentModel, experiment_id)
    if experiment is None:
        return None
    experiment.status = status
    if status == "failed" and experiment.decision is None:
        experiment.decision = "crash"
        experiment.decision_reason = "Training failed before producing a valid result."
    experiment.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(experiment)
    _refresh_run_summary(db, experiment.run_id)
    db.refresh(experiment)
    return _to_experiment_detail(experiment)


def get_experiment_config(db: Session, experiment_id: str) -> ExperimentConfig | None:
    """Load one experiment config."""
    experiment = db.get(ExperimentModel, experiment_id)
    if experiment is None:
        return None
    return ExperimentConfig.model_validate(experiment.experiment_config)


def _to_experiment_detail(experiment: ExperimentModel) -> ExperimentDetailResponse:
    return ExperimentDetailResponse(
        id=experiment.id,
        run_id=experiment.run_id,
        status=experiment.status,
        decision=experiment.decision,
        decision_reason=experiment.decision_reason,
        baseline_experiment_id=experiment.baseline_experiment_id,
        is_best_so_far=bool(experiment.is_best_so_far),
        config=ExperimentConfig.model_validate(experiment.experiment_config),
        parameter_space=EditableParameterSpace.model_validate(experiment.editable_parameter_space),
        proposal=ProposalSchema.model_validate(experiment.proposal) if experiment.proposal else None,
        result=ResultSchema.model_validate(experiment.result) if experiment.result else None,
        reflection=ReflectionSchema.model_validate(experiment.reflection) if experiment.reflection else None,
    )


def get_experiment_detail(db: Session, experiment_id: str) -> ExperimentDetailResponse | None:
    """Load one persisted experiment."""
    experiment = db.get(ExperimentModel, experiment_id)
    if experiment is None:
        return None
    return _to_experiment_detail(experiment)


def save_experiment_result(db: Session, experiment_id: str, result: ResultSchema) -> ExperimentDetailResponse | None:
    """Create or update the structured result for one experiment."""
    experiment = db.get(ExperimentModel, experiment_id)
    if experiment is None:
        return None

    stored_result = db.scalar(select(ResultModel).where(ResultModel.experiment_id == experiment_id))
    result_payload = result.model_dump()
    if stored_result is None:
        stored_result = ResultModel(
            id=f"result_{uuid4().hex[:8]}",
            experiment_id=experiment_id,
            status=result.status,
            metrics=result_payload["metrics"],
            resource=result_payload["resource"],
            params=result_payload["params"],
            artifacts=result_payload["artifacts"],
        )
        db.add(stored_result)
    else:
        stored_result.status = result.status
        stored_result.metrics = result_payload["metrics"]
        stored_result.resource = result_payload["resource"]
        stored_result.params = result_payload["params"]
        stored_result.artifacts = result_payload["artifacts"]
        stored_result.updated_at = datetime.utcnow()

    experiment.result = result_payload
    experiment.status = result.status
    experiment.updated_at = datetime.utcnow()
    db.commit()
    _refresh_run_summary(db, experiment.run_id)
    db.refresh(experiment)
    metrics = result_payload.get("metrics") or {}
    append_run_log(
        experiment.run_id,
        (
            f"[{experiment.id}] result saved | "
            f"top1_acc={metrics.get('top1_acc')} | "
            f"val_loss={metrics.get('val_loss')} | "
            f"train_loss={metrics.get('train_loss')} | "
            f"best_epoch={metrics.get('best_epoch')}"
        ),
    )
    return _to_experiment_detail(experiment)


def update_experiment_decision(
    db: Session,
    experiment_id: str,
    request: ExperimentDecisionRequest,
) -> ExperimentDetailResponse | None:
    """Persist one experiment research decision."""
    experiment = db.get(ExperimentModel, experiment_id)
    if experiment is None:
        return None
    experiment.decision = request.decision
    experiment.decision_reason = request.decision_reason
    experiment.updated_at = datetime.utcnow()
    db.commit()
    _refresh_run_summary(db, experiment.run_id)
    db.refresh(experiment)
    return _to_experiment_detail(experiment)


def get_run_summary(db: Session, run_id: str) -> RunSummaryResponse | None:
    """Return run-level research anchors and decision counts."""
    run = _refresh_run_summary(db, run_id)
    if run is None:
        return None
    experiments = db.scalars(select(ExperimentModel).where(ExperimentModel.run_id == run_id)).all()
    counts = {"keep": 0, "discard": 0, "crash": 0, "timeout": 0}
    for experiment in experiments:
        if experiment.decision in counts:
            counts[experiment.decision] += 1
    return RunSummaryResponse(
        run_id=run_id,
        baseline_experiment_id=run.baseline_experiment_id,
        best_experiment_id=run.best_experiment_id,
        frontier_experiment_id=run.frontier_experiment_id,
        keep_count=counts["keep"],
        discard_count=counts["discard"],
        crash_count=counts["crash"],
        timeout_count=counts["timeout"],
    )


def get_run_metrics(db: Session, run_id: str, metric_name: str) -> RunMetricsResponse | None:
    """Aggregate simple trend points from persisted results."""
    run = db.get(RunModel, run_id)
    if run is None:
        return None

    experiments = db.scalars(
        select(ExperimentModel).where(ExperimentModel.run_id == run_id).order_by(ExperimentModel.created_at.asc())
    ).all()
    points: list[PointMetric] = []
    available_metrics = {"train_loss", "val_loss", "top1_acc", "best_epoch"}
    for index, experiment in enumerate(experiments, start=1):
        result = db.scalar(select(ResultModel).where(ResultModel.experiment_id == experiment.id))
        if result is None:
            continue
        available_metrics.update(result.metrics.keys())
        metric_value = result.metrics.get(metric_name)
        if isinstance(metric_value, (int, float)):
            points.append(
                PointMetric(
                    experiment_id=experiment.id,
                    experiment_index=index,
                    metric_name=metric_name,
                    metric_value=float(metric_value),
                )
            )

    return RunMetricsResponse(
        run_id=run_id,
        metric_name=metric_name,
        available_metrics=sorted(available_metrics),
        points=points,
    )


def clear_all_records(db: Session) -> dict[str, int]:
    """Delete all persisted demo records."""
    deleted_results = db.query(ResultModel).delete()
    deleted_experiments = db.query(ExperimentModel).delete()
    deleted_runs = db.query(RunModel).delete()
    db.commit()
    return {
        "deleted_runs": deleted_runs,
        "deleted_experiments": deleted_experiments,
        "deleted_results": deleted_results,
    }


def clear_run_records(db: Session, run_id: str) -> dict[str, int] | None:
    """Delete one run and all of its experiments and results."""
    run = db.get(RunModel, run_id)
    if run is None:
        return None

    experiment_ids = [
        experiment.id
        for experiment in db.scalars(select(ExperimentModel).where(ExperimentModel.run_id == run_id)).all()
    ]
    deleted_results = 0
    if experiment_ids:
        deleted_results = (
            db.query(ResultModel)
            .filter(ResultModel.experiment_id.in_(experiment_ids))
            .delete(synchronize_session=False)
        )
    deleted_experiments = (
        db.query(ExperimentModel)
        .filter(ExperimentModel.run_id == run_id)
        .delete(synchronize_session=False)
    )
    db.delete(run)
    db.commit()
    return {
        "deleted_runs": 1,
        "deleted_experiments": deleted_experiments,
        "deleted_results": deleted_results,
    }


def discard_experiment(db: Session, experiment_id: str) -> ExperimentDetailResponse | None:
    """Mark one experiment discarded and remove partial artifacts/results."""
    experiment = db.get(ExperimentModel, experiment_id)
    if experiment is None:
        return None

    stored_result = db.scalar(select(ResultModel).where(ResultModel.experiment_id == experiment_id))
    if stored_result is not None:
        db.delete(stored_result)

    settings = get_settings()
    artifact_root = Path(settings.artifact_root)
    for artifact_path in (artifact_root / "checkpoints" / f"{experiment_id}.pt",):
        if artifact_path.exists():
            artifact_path.unlink()

    experiment.status = "discarded"
    experiment.decision = "discard"
    experiment.decision_reason = "Experiment was stopped or discarded before completion."
    experiment.result = None
    experiment.reflection = None
    experiment.updated_at = datetime.utcnow()
    db.commit()
    _refresh_run_summary(db, experiment.run_id)
    db.refresh(experiment)
    return _to_experiment_detail(experiment)
