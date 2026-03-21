"""FastAPI application entrypoint."""

from fastapi import FastAPI

from app.api.router import api_router
from app.core.settings import get_settings
from app.db.init_db import init_database


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="MVP backend for structured autonomous classification experiments.",
)
app.include_router(api_router)


@app.on_event("startup")
def initialize_database() -> None:
    """Create demo tables on application startup."""
    init_database()


@app.get("/health", tags=["system"])
def healthcheck() -> dict[str, str]:
    """Return a minimal health response."""
    return {"status": "ok"}
