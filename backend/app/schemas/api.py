"""Common API response envelope schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Generic, TypeVar

from pydantic import BaseModel, Field


ResponseDataT = TypeVar("ResponseDataT")


def _utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for API response metadata."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class ApiErrorDetail(BaseModel):
    """Structured API error detail."""

    message: str
    code: str | None = None
    field: str | None = None


class ApiResponseMeta(BaseModel):
    """Metadata attached to every API response envelope."""

    schema_version: str = "v1"
    timestamp: str = Field(default_factory=_utc_timestamp)


class ApiResponse(BaseModel, Generic[ResponseDataT]):
    """Stable JSON response envelope shared by API endpoints."""

    ok: bool
    code: str
    message: str
    data: ResponseDataT | None = None
    errors: list[ApiErrorDetail] = Field(default_factory=list)
    meta: ApiResponseMeta = Field(default_factory=ApiResponseMeta)

