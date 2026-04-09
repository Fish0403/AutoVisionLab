"""FastAPI application entrypoint."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.api.responses import build_success_response, register_exception_handlers
from app.core.settings import get_settings
from app.db.session import SessionLocal
from app.db.init_db import init_database
from app.schemas.api import ApiResponse
from app.services.task_store import cleanup_stale_task_payloads
from app.services.training_runner import cleanup_stale_running_experiments


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="MVP backend for structured autonomous classification experiments.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
register_exception_handlers(app)
app.include_router(api_router)


@app.on_event("startup")
def initialize_database() -> None:
    """Create demo tables on application startup."""
    init_database()
    cleanup_stale_task_payloads()
    db = SessionLocal()
    try:
        cleanup_stale_running_experiments(db)
    finally:
        db.close()


@app.get("/health", tags=["system"], response_model=ApiResponse[dict[str, str]])
def healthcheck() -> ApiResponse[dict[str, str]]:
    """Return a minimal health response."""
    return build_success_response({"status": "ok"}, message="Service is healthy.")
