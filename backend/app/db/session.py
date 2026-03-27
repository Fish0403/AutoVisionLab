"""Database engine and session factory helpers."""

from __future__ import annotations

from threading import Lock

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.settings import get_settings


_ENGINE_LOCK = Lock()
_ACTIVE_DATABASE_URL: str | None = None
_ACTIVE_ENGINE: Engine | None = None
_ACTIVE_SESSION_FACTORY: sessionmaker[Session] | None = None


def _build_engine(database_url: str) -> Engine:
    """Create one SQLAlchemy engine for the given database URL."""
    return create_engine(
        database_url,
        future=True,
        connect_args={"check_same_thread": False} if database_url.startswith("sqlite") else {},
    )


def get_engine() -> Engine:
    """Return an engine bound to the current configured database URL."""
    global _ACTIVE_DATABASE_URL, _ACTIVE_ENGINE, _ACTIVE_SESSION_FACTORY

    database_url = get_settings().database_url
    with _ENGINE_LOCK:
        if _ACTIVE_ENGINE is not None and _ACTIVE_DATABASE_URL == database_url:
            return _ACTIVE_ENGINE

        if _ACTIVE_ENGINE is not None:
            _ACTIVE_ENGINE.dispose()

        _ACTIVE_DATABASE_URL = database_url
        _ACTIVE_ENGINE = _build_engine(database_url)
        _ACTIVE_SESSION_FACTORY = sessionmaker(
            bind=_ACTIVE_ENGINE,
            autocommit=False,
            autoflush=False,
            future=True,
        )
        return _ACTIVE_ENGINE


def get_session_factory() -> sessionmaker[Session]:
    """Return a session factory bound to the current engine."""
    _ = get_engine()
    assert _ACTIVE_SESSION_FACTORY is not None
    return _ACTIVE_SESSION_FACTORY


def SessionLocal() -> Session:
    """Create one database session bound to the current engine."""
    return get_session_factory()()


def get_db_session():
    """Yield one database session for request-scoped usage."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
