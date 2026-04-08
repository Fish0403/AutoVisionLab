"""Helpers for consistent run.log message formatting."""

from __future__ import annotations

import json
from typing import Any, Literal


LogLevel = Literal["INFO", "WARNING", "ERROR"]


def _serialize_log_value(value: Any) -> str:
    """Return a compact log-safe string for one field value."""
    if isinstance(value, str):
        if any(char.isspace() for char in value) or any(char in value for char in "\";|"):
            return json.dumps(value, ensure_ascii=False)
        return value
    if isinstance(value, (list, dict, tuple, bool)) or value is None:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def format_run_log_message(
    *,
    level: LogLevel,
    section: str,
    message: str,
    **fields: Any,
) -> str:
    """Return one standardized run.log message body.

    Final line format after append_run_log():
    [YYYY-MM-DD HH:MM:SS] LEVEL |section| message; key=value; key=value
    """
    field_parts = [f"{key}={_serialize_log_value(value)}" for key, value in fields.items()]
    suffix = f"; {'; '.join(field_parts)}" if field_parts else ""
    return f"{level} |{section}| {message}{suffix}"
