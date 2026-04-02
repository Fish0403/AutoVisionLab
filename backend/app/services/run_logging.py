"""Run-level artifact logging helpers."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

from app.core.settings import get_settings


def get_artifact_root() -> Path:
    """Return the configured artifact root path."""
    settings = get_settings()
    artifact_root = Path(settings.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    return artifact_root


def get_run_artifact_dir(run_id: str) -> Path:
    """Return the canonical artifact directory for one run."""
    run_dir = get_artifact_root() / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_experiment_artifact_dir(run_id: str, experiment_id: str) -> Path:
    """Return the canonical artifact directory for one experiment."""
    experiment_dir = get_run_artifact_dir(run_id) / "experiments" / experiment_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def get_run_log_path(run_id: str) -> Path:
    """Return the canonical log path for one run."""
    return get_run_artifact_dir(run_id) / "run.log"


def get_run_llm_log_path(run_id: str) -> Path:
    """Return the canonical LLM JSONL log path for one run."""
    return get_run_artifact_dir(run_id) / "llm.jsonl"


def get_experiment_checkpoint_path(run_id: str, experiment_id: str) -> Path:
    """Return the canonical checkpoint path for one experiment."""
    return get_experiment_artifact_dir(run_id, experiment_id) / "checkpoint.pt"


def get_experiment_recipe_path(run_id: str, experiment_id: str) -> Path:
    """Return the canonical recipe snapshot path for one experiment."""
    return get_experiment_artifact_dir(run_id, experiment_id) / "recipe.json"


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
