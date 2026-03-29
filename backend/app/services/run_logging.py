"""Run-level artifact logging helpers."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from app.core.settings import get_settings


def get_run_log_path(run_id: str) -> Path:
    """Return the canonical log path for one run."""
    settings = get_settings()
    log_path = Path(settings.artifact_root) / "runs" / f"{run_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


def get_run_llm_log_path(run_id: str) -> Path:
    """Return the canonical LLM JSONL log path for one run."""
    settings = get_settings()
    log_path = Path(settings.artifact_root) / "runs" / f"{run_id}.llm.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return log_path


def append_run_log(run_id: str, message: str) -> Path:
    """Append one timestamped line to the run log."""
    log_path = get_run_log_path(run_id)
    timestamp = datetime.utcnow().isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] {message}\n")
    return log_path


def append_run_llm_event(run_id: str, event_type: str, payload: dict[str, Any]) -> Path:
    """Append one structured LLM event to the run-level JSONL log."""
    log_path = get_run_llm_log_path(run_id)
    event_payload = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "event_type": event_type,
        "payload": payload,
    }
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(event_payload, ensure_ascii=False) + "\n")
    return log_path
