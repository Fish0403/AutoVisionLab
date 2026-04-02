"""Shared schema definitions."""

from typing import Literal

from pydantic import BaseModel, Field


TaskType = Literal["classification"]
ModelFamily = Literal["mobilenet", "googlenet", "resnet"]
ModelName = Literal[
    "mobilenet_v2",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "efficientnet_b0",
    "efficientnet_b1",
    "googlenet",
    "resnet18",
    "resnet34",
    "resnet50",
]
RiskLevel = Literal["low", "medium", "high"]
ExperimentStatus = Literal["draft", "queued", "running", "success", "failed", "discarded"]
RunStatus = Literal["draft", "active", "paused", "completed", "failed"]
ReflectionOutcome = Literal["improved", "neutral", "degraded", "failed"]
ResultStatus = Literal["success", "failed"]
ExperimentDecision = Literal["keep", "discard", "crash", "timeout"]


class ArtifactPaths(BaseModel):
    """Artifact locations produced by training."""

    log_path: str
    checkpoint_path: str
    recipe_path: str | None = None


class ResourceUsage(BaseModel):
    """Training resource consumption."""

    gpu_memory_mb: int | None = None
    training_seconds: int | None = None
    latency_ms: float | None = Field(default=None, ge=0)
    parameter_count_million: float | None = Field(default=None, ge=0)


class MetricsSnapshot(BaseModel):
    """Structured scalar metrics for one experiment."""

    train_loss: float | None = None
    val_loss: float | None = None
    top1_acc: float | None = None
    best_epoch: int | None = None


class PointMetric(BaseModel):
    """Trend point used by the frontend chart."""

    experiment_id: str
    experiment_index: int = Field(ge=1)
    metric_name: str
    metric_value: float
