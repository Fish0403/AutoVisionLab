"""Helpers for building consistent API response envelopes."""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.schemas.api import ApiErrorDetail, ApiResponse


def build_success_response(data: Any, *, message: str, code: str = "success") -> ApiResponse[Any]:
    """Build a successful API response envelope."""
    return ApiResponse(ok=True, code=code, message=message, data=data)


def build_error_response(
    *,
    status_code: int,
    message: str,
    code: str,
    errors: list[ApiErrorDetail] | None = None,
) -> JSONResponse:
    """Build an error API response envelope."""
    payload = ApiResponse[None](
        ok=False,
        code=code,
        message=message,
        data=None,
        errors=errors or [ApiErrorDetail(message=message, code=code)],
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def register_exception_handlers(app) -> None:
    """Register shared exception handlers on the FastAPI application."""

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        message = exc.detail if isinstance(exc.detail, str) else "Request failed."
        if exc.status_code == status.HTTP_400_BAD_REQUEST:
            code = "bad_request"
        elif exc.status_code == status.HTTP_404_NOT_FOUND:
            code = "not_found"
        elif exc.status_code == status.HTTP_409_CONFLICT:
            code = "conflict"
        elif exc.status_code == status.HTTP_502_BAD_GATEWAY:
            code = "upstream_error"
        else:
            code = "request_error"
        return build_error_response(status_code=exc.status_code, message=message, code=code)

    @app.exception_handler(RequestValidationError)
    async def _validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [
            ApiErrorDetail(
                message=error["msg"],
                code="validation_error",
                field=".".join(str(part) for part in error["loc"]),
            )
            for error in exc.errors()
        ]
        return build_error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            message="Request validation failed.",
            code="validation_error",
            errors=errors,
        )

    @app.exception_handler(Exception)
    async def _unexpected_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        return build_error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=str(exc) or "Internal server error.",
            code="internal_error",
        )
