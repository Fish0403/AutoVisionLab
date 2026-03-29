"""Ranking policy schemas."""

from typing import Literal

from pydantic import BaseModel, Field


RankingMetric = Literal["top1_acc", "val_loss", "training_seconds", "latency_ms", "parameter_count_million"]
RankingMetricMode = Literal["max", "min"]


class RankingPolicy(BaseModel):
    """Configurable experiment ranking policy for one run."""

    primary_metric: RankingMetric = "top1_acc"
    primary_metric_mode: RankingMetricMode = "max"
    min_primary_metric_improvement: float = Field(default=0.01, ge=0)
    primary_metric_parity_epsilon: float = Field(default=0.0005, ge=0)
    tie_breaker_metric: RankingMetric = "val_loss"
    tie_breaker_mode: RankingMetricMode = "min"
    min_tie_breaker_metric_improvement: float = Field(default=0.01, ge=0)
    max_image_size: int | None = Field(default=None, gt=0)
