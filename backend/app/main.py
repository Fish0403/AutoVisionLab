"""FastAPI application entrypoint."""

from fastapi import FastAPI

from app.api.router import api_router
from app.api.responses import register_exception_handlers
from app.core.settings import get_settings
from app.db.session import SessionLocal
from app.db.init_db import init_database
from app.services.training_runner import cleanup_stale_running_experiments


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="MVP backend for structured autonomous classification experiments.",
)
register_exception_handlers(app)
app.include_router(api_router)


@app.on_event("startup")
def initialize_database() -> None:
    """Create demo tables on application startup."""
    init_database()
    db = SessionLocal()
    try:
        cleanup_stale_running_experiments(db)
    finally:
        db.close()


@app.get("/health", tags=["system"])
def healthcheck() -> dict[str, str]:
    """Return a minimal health response."""
    return {"status": "ok"}
