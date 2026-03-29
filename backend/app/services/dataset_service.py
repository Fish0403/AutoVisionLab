"""Helpers for discovering local datasets."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from app.core.settings import get_settings
from app.schemas.dataset import LocalDatasetSummary
from app.services.dataset_paths import (
    normalize_dataset_name,
    resolve_classification_dataset_layout,
)


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


def list_local_datasets() -> list[LocalDatasetSummary]:
    """List locally discovered datasets and their training readiness."""
    settings = get_settings()
    data_root = Path(settings.data_root)
    raw_root = data_root / "raw"
    classification_root = data_root / "classification"

    dataset_names = collect_dataset_names(raw_root=raw_root, classification_root=classification_root)
    dataset_summaries = [build_local_dataset_summary(data_root=data_root, dataset_name=name) for name in dataset_names]
    return sorted(dataset_summaries, key=lambda item: (not item.is_ready_for_training, item.name.lower()))


def collect_dataset_names(raw_root: Path, classification_root: Path) -> list[str]:
    """Collect dataset names from raw and classification directories."""
    dataset_names: list[str] = []
    if raw_root.exists():
        dataset_names.extend(path.name for path in raw_root.iterdir() if path.is_dir())
    if classification_root.exists():
        classification_names = [path.name for path in classification_root.iterdir() if path.is_dir()]
        dataset_names = classification_names + [name for name in dataset_names if name not in classification_names]
    deduped_names: list[str] = []
    seen_normalized_names: set[str] = set()
    for dataset_name in dataset_names:
        normalized_name = normalize_dataset_name(dataset_name)
        if normalized_name in seen_normalized_names:
            continue
        deduped_names.append(dataset_name)
        seen_normalized_names.add(normalized_name)
    return deduped_names


def build_local_dataset_summary(data_root: Path, dataset_name: str) -> LocalDatasetSummary:
    """Build one dataset readiness snapshot."""
    raw_dir, prepared_source_dir, source_dir, classification_dir = resolve_classification_dataset_layout(
        data_root,
        dataset_name,
    )
    train_manifest = classification_dir / "train.txt"
    val_manifest = classification_dir / "val.txt"
    test_manifest = classification_dir / "test.txt"

    has_source_dir = source_dir.exists() and source_dir.is_dir()
    is_ready_for_training = has_source_dir and train_manifest.exists() and val_manifest.exists()
    original_image_size = detect_original_image_size(source_dir) if has_source_dir else None
    image_size_options = build_image_size_options(original_image_size)

    if is_ready_for_training:
        message = "Ready for training."
    elif raw_dir.exists() and not prepared_source_dir.exists() and (raw_dir / "NEU-CLS").exists():
        message = "Raw source exists, but preprocessing is still required."
    elif has_source_dir and not train_manifest.exists():
        message = "Source exists, but train.txt is missing."
    elif has_source_dir and not val_manifest.exists():
        message = "Source exists, but val.txt is missing."
    elif classification_dir.exists():
        message = "Classification manifests are incomplete."
    elif raw_dir.exists():
        message = "Raw source exists, but classification manifests are missing."
    else:
        message = "Dataset directories were not found."

    return LocalDatasetSummary(
        name=dataset_name,
        source_dir=str(source_dir) if source_dir.exists() else None,
        classification_dir=str(classification_dir) if classification_dir.exists() else None,
        has_source_dir=has_source_dir,
        has_prepared_source_dir=prepared_source_dir.exists(),
        train_manifest_exists=train_manifest.exists(),
        val_manifest_exists=val_manifest.exists(),
        test_manifest_exists=test_manifest.exists(),
        is_ready_for_training=is_ready_for_training,
        original_image_size=original_image_size,
        image_size_options=image_size_options,
        message=message,
    )


def detect_original_image_size(source_dir: Path) -> int | None:
    """Detect one representative image size from the source directory."""
    image_path = find_first_image(source_dir)
    if image_path is None:
        return None
    with Image.open(image_path) as image:
        width, height = image.size
    return max(width, height)


def find_first_image(source_dir: Path) -> Path | None:
    """Find the first supported image file under one source root."""
    for path in source_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            return path
    return None


def build_image_size_options(original_image_size: int | None) -> list[int]:
    """Build suggested image size options for the UI."""
    if original_image_size is None:
        return []

    candidate_sizes = [original_image_size]
    standard_sizes = [64, 96, 128, 160, 192, 224, 256]
    for size in standard_sizes:
        if size >= original_image_size and size not in candidate_sizes:
            candidate_sizes.append(size)
        if len(candidate_sizes) >= 3:
            break
    return candidate_sizes
