"""Filesystem loaders for model manifest YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml

from app.model_catalog.schemas import ModelManifest


MODEL_MANIFEST_ROOT = Path(__file__).resolve().parent.parent / "model_manifests"


def iter_manifest_paths(*, task_type: str = "classification") -> list[Path]:
    """Return all manifest file paths for one task type."""
    manifest_dir = MODEL_MANIFEST_ROOT / task_type
    return sorted(manifest_dir.glob("*.yaml"))


def load_manifest_from_path(path: Path) -> ModelManifest:
    """Load one manifest YAML file."""
    with path.open("r", encoding="utf-8") as manifest_file:
        payload = yaml.safe_load(manifest_file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Model manifest must be a mapping: {path}")
    return ModelManifest.model_validate(payload)

