"""Persistence helpers for background workspace task snapshots."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select

from app.db.session import SessionLocal
from app.models.task import BackgroundTaskModel


ACTIVE_TASK_STATUSES = frozenset({"queued", "running", "stopping"})
TASK_RESTART_STOP_REASON = "Task interrupted by backend restart."


def _now_utc() -> datetime:
    """Return the current timezone-aware UTC time."""
    return datetime.now(timezone.utc)


def _parse_iso_timestamp(timestamp_text: str | None) -> datetime | None:
    """Parse one ISO timestamp string when possible."""
    if not timestamp_text:
        return None
    try:
        return datetime.fromisoformat(timestamp_text)
    except ValueError:
        return None


def upsert_task_payload(task_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Persist one task snapshot and return the normalized payload."""
    task_id = str(payload.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("Task payload requires a task_id")

    normalized_payload = deepcopy(payload)
    created_at = _parse_iso_timestamp(normalized_payload.get("created_at")) or _now_utc()
    updated_at = _parse_iso_timestamp(normalized_payload.get("updated_at")) or created_at
    normalized_payload["created_at"] = created_at.isoformat()
    normalized_payload["updated_at"] = updated_at.isoformat()

    with SessionLocal() as db:
        stored_task = db.get(BackgroundTaskModel, task_id)
        if stored_task is None:
            stored_task = BackgroundTaskModel(
                task_id=task_id,
                task_type=task_type,
                status=str(normalized_payload.get("status") or "unknown"),
                payload=normalized_payload,
                created_at=created_at,
                updated_at=updated_at,
            )
        else:
            stored_task.task_type = task_type
            stored_task.status = str(normalized_payload.get("status") or stored_task.status)
            stored_task.payload = normalized_payload
            stored_task.created_at = created_at
            stored_task.updated_at = updated_at
        db.add(stored_task)
        db.commit()
    return normalized_payload


def get_task_payload(task_type: str, task_id: str) -> dict[str, Any] | None:
    """Load one persisted task snapshot by identifier."""
    with SessionLocal() as db:
        stored_task = db.get(BackgroundTaskModel, task_id)
        if stored_task is None or stored_task.task_type != task_type:
            return None
        return deepcopy(stored_task.payload)


def delete_task_payload(task_type: str, task_id: str) -> bool:
    """Delete one persisted task snapshot by identifier."""
    with SessionLocal() as db:
        stored_task = db.get(BackgroundTaskModel, task_id)
        if stored_task is None or stored_task.task_type != task_type:
            return False
        db.delete(stored_task)
        db.commit()
        return True


def list_task_payloads(task_type: str) -> list[dict[str, Any]]:
    """Return all persisted task snapshots of the given type."""
    with SessionLocal() as db:
        stored_tasks = db.scalars(
            select(BackgroundTaskModel)
            .where(BackgroundTaskModel.task_type == task_type)
            .order_by(BackgroundTaskModel.updated_at.desc())
        ).all()
        return [deepcopy(task.payload) for task in stored_tasks]


def get_active_task_payload(task_type: str) -> dict[str, Any] | None:
    """Return the newest persisted active task snapshot of the given type."""
    with SessionLocal() as db:
        stored_task = db.scalar(
            select(BackgroundTaskModel)
            .where(
                BackgroundTaskModel.task_type == task_type,
                BackgroundTaskModel.status.in_(tuple(ACTIVE_TASK_STATUSES)),
            )
            .order_by(BackgroundTaskModel.updated_at.desc())
        )
        if stored_task is None:
            return None
        return deepcopy(stored_task.payload)


def cleanup_stale_task_payloads() -> int:
    """Mark tasks left active by a previous backend process as stopped."""
    with SessionLocal() as db:
        stale_tasks = db.scalars(
            select(BackgroundTaskModel).where(BackgroundTaskModel.status.in_(tuple(ACTIVE_TASK_STATUSES)))
        ).all()
        if not stale_tasks:
            return 0

        stopped_at = _now_utc()
        stopped_at_text = stopped_at.isoformat()
        for stored_task in stale_tasks:
            payload = deepcopy(stored_task.payload)
            payload["status"] = "stopped"
            payload["stop_requested"] = True
            payload["stop_reason"] = payload.get("stop_reason") or TASK_RESTART_STOP_REASON
            payload["updated_at"] = stopped_at_text
            stored_task.status = "stopped"
            stored_task.payload = payload
            stored_task.updated_at = stopped_at
        db.commit()
        return len(stale_tasks)
