"""Database initialization helpers for the demo setup."""

from app.db.base import Base
from app.db.session import engine
from app.models import ExperimentModel, ResultModel, RunModel


def init_database() -> None:
    """Create all declared tables for the demo environment."""
    _ = (RunModel, ExperimentModel, ResultModel)
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    init_database()
    print("Initialized database tables.")

