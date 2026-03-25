"""Static catalog service used by the frontend skeleton."""

from app.schemas.ai import ProposalSchema, ReflectionSchema, ResultSchema
from app.schemas.common import ArtifactPaths, MetricsSnapshot, PointMetric, ResourceUsage
from app.schemas.experiment import ExperimentDetailResponse, ExperimentSummary
from app.schemas.parameter_space import ExperimentConfig, ExperimentParams, SearchPolicy
from app.schemas.run import RunDetailResponse, RunListItem, RunMetricsResponse
from app.services.parameter_space import get_parameter_space


def _sample_config() -> ExperimentConfig:
    return ExperimentConfig(
        task_type="classification",
        dataset="cifar10",
        model_family="mobilenet",
        model_name="mobilenet_v2",
        parameter_space_version="mobilenet_v2@v1",
        search_policy=SearchPolicy(),
        params=ExperimentParams(
            optimizer="adamw",
            learning_rate=0.003,
            batch_size=128,
            image_size=64,
            epochs=30,
            weight_decay=0.0001,
            scheduler="cosine",
            augmentation_policy="basic",
            loss_name="cross_entropy_with_label_smoothing",
            label_smoothing=0.1,
        ),
    )


def list_runs() -> list[RunListItem]:
    """Return seed data for the initial UI."""
    return [
        RunListItem(
            id="run_demo_001",
            name="CIFAR-10 MobileNet baseline tuning",
            dataset="cifar10",
            model_name="mobilenet_v2",
            status="active",
            experiment_count=3,
        )
    ]


def get_run_detail(run_id: str) -> RunDetailResponse | None:
    """Return one demo run."""
    if run_id != "run_demo_001":
        return None
    return RunDetailResponse(
        id=run_id,
        name="CIFAR-10 MobileNet baseline tuning",
        dataset="cifar10",
        model_name="mobilenet_v2",
        status="active",
        experiments=[
            ExperimentSummary(id="exp_demo_001", status="success", model_name="mobilenet_v2"),
            ExperimentSummary(id="exp_demo_002", status="success", model_name="mobilenet_v2"),
            ExperimentSummary(id="exp_demo_003", status="running", model_name="mobilenet_v2"),
        ],
    )


def get_run_metrics(run_id: str, metric_name: str) -> RunMetricsResponse | None:
    """Return demo trend points for one run."""
    if run_id != "run_demo_001":
        return None
    points = [
        PointMetric(experiment_id="exp_demo_001", experiment_index=1, metric_name=metric_name, metric_value=0.71),
        PointMetric(experiment_id="exp_demo_002", experiment_index=2, metric_name=metric_name, metric_value=0.78),
        PointMetric(experiment_id="exp_demo_003", experiment_index=3, metric_name=metric_name, metric_value=0.80),
    ]
    return RunMetricsResponse(
        run_id=run_id,
        metric_name=metric_name,
        available_metrics=["top1_acc", "val_loss", "train_loss"],
        points=points,
    )


def get_experiment_detail(experiment_id: str) -> ExperimentDetailResponse | None:
    """Return one demo experiment payload."""
    if experiment_id not in {"exp_demo_001", "exp_demo_002", "exp_demo_003"}:
        return None
    return ExperimentDetailResponse(
        id=experiment_id,
        run_id="run_demo_001",
        status="success" if experiment_id != "exp_demo_003" else "running",
        config=_sample_config(),
        parameter_space=get_parameter_space("mobilenet_v2"),
        proposal=ProposalSchema(
            model_name="mobilenet_v2",
            based_on_experiment_ids=["exp_demo_001", "exp_demo_002"],
            hypothesis="A slightly higher learning rate may improve early convergence.",
            changes={"learning_rate": 0.004, "label_smoothing": 0.08},
            reason="Recent runs still look mildly underfit.",
            risk="low",
        ),
        result=ResultSchema(
            status="success",
            metrics=MetricsSnapshot(train_loss=0.42, val_loss=0.51, top1_acc=0.78, best_epoch=24),
            resource=ResourceUsage(gpu_memory_mb=2100, training_seconds=320),
            params=_sample_config().params,
            artifacts=ArtifactPaths(
                log_path="artifacts/logs/exp_demo_002.log",
                checkpoint_path="artifacts/checkpoints/exp_demo_002.pt",
            ),
        ),
        reflection=ReflectionSchema(
            outcome="improved",
            analysis="Higher learning rate improved convergence without visible instability.",
            confidence=0.78,
            next_action="Explore nearby learning rates while keeping cosine scheduler.",
            recommended_changes={"learning_rate": 0.005},
        ),
    )
