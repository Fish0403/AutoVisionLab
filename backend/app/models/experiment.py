"""Experiment ORM model."""

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ExperimentModel(Base):
    """Persist a structured experiment snapshot."""

    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    decision: Mapped[str | None] = mapped_column(String(32), nullable=True)
    decision_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    baseline_experiment_id: Mapped[str | None] = mapped_column(ForeignKey("experiments.id"), nullable=True)
    is_best_so_far: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    experiment_config: Mapped[dict] = mapped_column(JSON, nullable=False)
    editable_parameter_space: Mapped[dict] = mapped_column(JSON, nullable=False)
    proposal: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    reflection: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
