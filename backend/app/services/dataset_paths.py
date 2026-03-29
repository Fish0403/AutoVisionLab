"""Shared dataset path resolution helpers."""

from __future__ import annotations

from pathlib import Path


PREPARED_SOURCE_DIRNAME = "classification_source"


def normalize_dataset_name(dataset_name: str) -> str:
    """Normalize one dataset name for tolerant directory lookup."""
    return dataset_name.strip().lower().replace("_", "-")


def resolve_dataset_dir(parent_dir: Path, dataset_name: str, *, strict: bool = False) -> Path:
    """Resolve one dataset directory with alias and case-insensitive fallback."""
    direct_dir = parent_dir / dataset_name
    if direct_dir.exists():
        return direct_dir

    if not parent_dir.exists():
        if strict:
            raise FileNotFoundError(f"Dataset parent directory not found: {parent_dir}")
        return direct_dir

    normalized_name = normalize_dataset_name(dataset_name)
    for child_dir in sorted(path for path in parent_dir.iterdir() if path.is_dir()):
        if normalize_dataset_name(child_dir.name) == normalized_name:
            return child_dir

    if strict:
        raise FileNotFoundError(
            f"Dataset directory not found under {parent_dir} for dataset={dataset_name!r}"
        )
    return direct_dir


def resolve_classification_dataset_layout(data_root: Path, dataset_name: str) -> tuple[Path, Path, Path, Path]:
    """Resolve raw, prepared, source, and classification directories for one dataset."""
    raw_dir = resolve_dataset_dir(data_root / "raw", dataset_name)
    prepared_source_dir = raw_dir / PREPARED_SOURCE_DIRNAME
    source_dir = prepared_source_dir if prepared_source_dir.exists() else raw_dir
    classification_dir = resolve_dataset_dir(data_root / "classification", dataset_name)
    return raw_dir, prepared_source_dir, source_dir, classification_dir
