"""Database initialization helpers for the demo setup."""

from sqlalchemy import inspect, text

from app.db.base import Base
from app.db.session import engine
from app.models import ExperimentModel, ResultModel, RunModel


def _ensure_column(table_name: str, column_name: str, ddl: str) -> None:
    inspector = inspect(engine)
    columns = {column["name"] for column in inspector.get_columns(table_name)}
    if column_name in columns:
        return
    with engine.begin() as connection:
        connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {ddl}"))


def _migrate_demo_schema() -> None:
    _ensure_column("runs", "baseline_experiment_id", "baseline_experiment_id VARCHAR(64)")
    _ensure_column("runs", "best_experiment_id", "best_experiment_id VARCHAR(64)")
    _ensure_column("runs", "frontier_experiment_id", "frontier_experiment_id VARCHAR(64)")
    _ensure_column("experiments", "decision", "decision VARCHAR(32)")
    _ensure_column("experiments", "decision_reason", "decision_reason TEXT")
    _ensure_column("experiments", "baseline_experiment_id", "baseline_experiment_id VARCHAR(64)")
    _ensure_column("experiments", "is_best_so_far", "is_best_so_far BOOLEAN NOT NULL DEFAULT 0")


def init_database() -> None:
    """Create all declared tables for the demo environment."""
    _ = (RunModel, ExperimentModel, ResultModel)
    Base.metadata.create_all(bind=engine)
    _migrate_demo_schema()


if __name__ == "__main__":
    init_database()
    print("Initialized database tables.")
