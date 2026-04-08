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


def get_run_prompt_context_log_path(run_id: str) -> Path:
    """Return the canonical prompt-context JSON path for one run."""
    return get_run_artifact_dir(run_id) / "prompt_context.json"


def get_run_prompt_markdown_path(run_id: str) -> Path:
    """Return the canonical human-readable prompt log path for one run."""
    return get_run_artifact_dir(run_id) / "proposal_prompts.md"


def get_experiment_checkpoint_path(run_id: str, experiment_id: str) -> Path:
    """Return the canonical checkpoint path for one experiment."""
    return get_experiment_artifact_dir(run_id, experiment_id) / "checkpoint.pt"


def get_experiment_recipe_path(run_id: str, experiment_id: str) -> Path:
    """Return the canonical recipe snapshot path for one experiment."""
    return get_experiment_artifact_dir(run_id, experiment_id) / "recipe.json"


def append_run_log(run_id: str, message: str) -> Path:
    """Append one timestamped line to the run log."""
    log_path = get_run_log_path(run_id)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
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


def append_run_prompt_context_event(run_id: str, event_type: str, payload: dict[str, Any]) -> Path:
    """Append one structured prompt-context event to the run-level JSON log."""
    log_path = get_run_prompt_context_log_path(run_id)
    event_payload = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "event_type": event_type,
        "payload": payload,
    }
    if log_path.exists():
        with log_path.open(encoding="utf-8") as log_file:
            log_payload = json.load(log_file)
    else:
        log_payload = {
            "run_id": run_id,
            "events": [],
        }
    log_payload.setdefault("run_id", run_id)
    log_payload.setdefault("events", [])
    log_payload["events"].append(event_payload)
    with log_path.open("w", encoding="utf-8") as log_file:
        json.dump(log_payload, log_file, ensure_ascii=False, indent=2)
        log_file.write("\n")
    return log_path


def append_run_prompt_markdown_event(run_id: str, event_type: str, payload: dict[str, Any]) -> Path:
    """Append one human-readable prompt event to the run-level Markdown log."""
    log_path = get_run_prompt_markdown_path(run_id)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    if not log_path.exists():
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("# Proposal Prompt Log\n\n")
            log_file.write("This file is a human-readable view of proposal prompt requests and responses.\n\n")

    attempt = payload.get("attempt")
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"## [{timestamp}] {event_type}\n\n")
        if attempt is not None:
            log_file.write(f"- Attempt: `{attempt}`\n")
        for key in ("history_items", "prompt_chars", "prompt_tokens_estimate", "response_model", "response_chars"):
            if payload.get(key) is not None:
                log_file.write(f"- {key}: `{payload[key]}`\n")
        usage = payload.get("usage")
        if usage is not None:
            log_file.write(f"- usage: `{json.dumps(usage, ensure_ascii=False)}`\n")
        error = payload.get("error")
        if error is not None:
            log_file.write(f"- error: `{error}`\n")
        log_file.write("\n")

        system_prompt = payload.get("system_prompt")
        if system_prompt is not None:
            log_file.write("### System Prompt\n\n```text\n")
            log_file.write(f"{system_prompt}\n")
            log_file.write("```\n\n")

        user_prompt = payload.get("user_prompt")
        if user_prompt is not None:
            log_file.write("### User Prompt\n\n```text\n")
            log_file.write(f"{user_prompt}\n")
            log_file.write("```\n\n")

        raw_content = payload.get("raw_content")
        if raw_content is not None:
            log_file.write("### Raw Content\n\n```text\n")
            log_file.write(f"{raw_content}\n")
            log_file.write("```\n\n")

        parsed_payload = payload.get("parsed_payload")
        if parsed_payload is not None:
            log_file.write("### Parsed Payload\n\n```json\n")
            log_file.write(f"{json.dumps(parsed_payload, ensure_ascii=False, indent=2)}\n")
            log_file.write("```\n\n")
    return log_path
