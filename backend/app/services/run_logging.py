"""Run-level artifact logging helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from app.core.settings import get_settings


def get_run_log_path(run_id: str) -> Path:
    """Return the canonical log path for one run."""
    settings = get_settings()
    log_path = Path(settings.artifact_root) / "runs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


def append_run_log(run_id: str, message: str) -> Path:
    """Append one timestamped line to the run log."""
    log_path = get_run_log_path(run_id)
    timestamp = datetime.utcnow().isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] {message}\n")
    return log_path
