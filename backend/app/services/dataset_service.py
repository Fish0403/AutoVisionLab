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


KNOWN_DATASET_IMAGE_SPECS: dict[str, tuple[int, int, list[int]]] = {
    "neu": (200, 200, [200, 224, 256]),
}

def list_local_datasets() -> list[LocalDatasetSummary]:
    """List locally discovered datasets and their training readiness."""
    settings = get_settings()
    data_root = Path(settings.data_root)
    raw_root = data_root / "raw"
    classification_root = data_root / "classification"

    dataset_names = collect_dataset_names(raw_root=raw_root, classification_root=classification_root)
    dataset_summaries = [build_local_dataset_summary(data_root=data_root, dataset_name=name) for name in dataset_names]
    return sorted(dataset_summaries, key=lambda item: (not item.is_ready_for_training, item.name.lower()))


def get_local_dataset_summary(dataset_name: str) -> LocalDatasetSummary | None:
    """Return one dataset summary by name when it can be discovered locally."""
    settings = get_settings()
    data_root = Path(settings.data_root)
    raw_root = data_root / "raw"
    classification_root = data_root / "classification"
    for candidate_name in collect_dataset_names(raw_root=raw_root, classification_root=classification_root):
        if normalize_dataset_name(candidate_name) == normalize_dataset_name(dataset_name):
            return build_local_dataset_summary(data_root=data_root, dataset_name=candidate_name)
    return None


def build_dataset_summary_text(
    dataset_summary: LocalDatasetSummary | None,
    training_image_size: int | None = None,
) -> str | None:
    """Build one compact dataset summary line for the workspace UI."""
    if dataset_summary is None:
        return None

    parts: list[str] = []
    class_count = len(dataset_summary.class_names)
    if class_count > 0:
        parts.append(f"{class_count} classes")
    parts.append(f"train {int(dataset_summary.train_sample_count)}")
    parts.append(f"val {int(dataset_summary.val_sample_count)}")
    parts.append(f"test {int(dataset_summary.test_sample_count)}")
    if dataset_summary.image_width and dataset_summary.image_height:
        parts.append(f"original {dataset_summary.image_width}x{dataset_summary.image_height}")
    if training_image_size and training_image_size > 0:
        parts.append(f"train size {training_image_size}x{training_image_size}")
    return " · ".join(parts) if parts else None


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
    normalized_dataset_name = normalize_dataset_name(dataset_name)
    known_width, known_height, known_image_size_options = KNOWN_DATASET_IMAGE_SPECS.get(normalized_dataset_name, (None, None, []))
    train_class_distribution = load_manifest_class_distribution(train_manifest)
    val_class_distribution = load_manifest_class_distribution(val_manifest)
    test_class_distribution = load_manifest_class_distribution(test_manifest)
    class_names = sorted(set(train_class_distribution) | set(val_class_distribution) | set(test_class_distribution))
    train_sample_count = sum(train_class_distribution.values())
    val_sample_count = sum(val_class_distribution.values())
    test_sample_count = sum(test_class_distribution.values())
    image_width = known_width
    image_height = known_height
    if image_width is None or image_height is None:
        image_width, image_height = detect_manifest_image_size(train_manifest, source_dir) or (None, None)
    original_image_size = max(image_width, image_height) if image_width and image_height else None
    image_size_options = list(known_image_size_options)

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
        image_width=image_width,
        image_height=image_height,
        image_size_options=image_size_options,
        train_sample_count=train_sample_count,
        val_sample_count=val_sample_count,
        test_sample_count=test_sample_count,
        class_names=class_names,
        train_class_distribution=train_class_distribution,
        val_class_distribution=val_class_distribution,
        test_class_distribution=test_class_distribution,
        message=message,
    )


def load_manifest_class_distribution(manifest_path: Path) -> dict[str, int]:
    """Load per-class sample counts from one manifest when it exists."""
    if not manifest_path.exists():
        return {}

    distribution: dict[str, int] = {}
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "\t" in line:
            _, class_name = line.rsplit("\t", maxsplit=1)
        else:
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            _, class_name = parts
        distribution[class_name] = distribution.get(class_name, 0) + 1
    return dict(sorted(distribution.items()))


def detect_manifest_image_size(manifest_path: Path, source_root: Path) -> tuple[int, int] | None:
    """Read one image size from the first valid manifest entry."""
    if not manifest_path.exists():
        return None

    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "\t" in line:
            relative_path_text, _ = line.rsplit("\t", maxsplit=1)
        else:
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            relative_path_text, _ = parts
        image_path = source_root / Path(relative_path_text)
        if not image_path.exists():
            continue
        with Image.open(image_path) as image:
            return image.size
    return None
