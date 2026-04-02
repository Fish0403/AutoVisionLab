"""Prepare the flat NEU-CLS dataset into split manifests.

Expected source layout:

data/raw/NEU-CLS/
  Cr_161.bmp
  In_001.bmp
  ...

Output layout:

data/classification/<dataset_name>/
  train.txt
  val.txt
  test.txt
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
DEFAULT_VAL_RATIO = 0.2
DEFAULT_TEST_RATIO = 0.1
DEFAULT_SEED = 42
DATA_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = DATA_ROOT / "raw" / "NEU-CLS"
DEFAULT_DATASET_NAME = "NEU"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Split a flat NEU-CLS dataset into train/val/test manifest files."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Extracted NEU-CLS root directory. Default: %(default)s",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="Base data directory that contains raw/ and classification/. Default: %(default)s",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_DATASET_NAME,
        help="Dataset name used for the output directory. Default: %(default)s",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Validation ratio per class. Default: %(default)s",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_TEST_RATIO,
        help="Test ratio per class. Default: %(default)s",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for the split. Default: %(default)s",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove the output directory first if it already exists.",
    )
    return parser.parse_args()


def infer_class_name(image_path: Path) -> str:
    """Infer the class name from the file name prefix."""
    stem = image_path.stem
    parts = stem.rsplit("_", maxsplit=1)
    if len(parts) != 2 or not parts[1].isdigit():
        raise ValueError(f"Unexpected NEU-CLS file name format: {image_path.name}")
    return parts[0]


def collect_images_by_class(source_root: Path) -> dict[str, list[Path]]:
    """Collect all supported images grouped by class name."""
    if not source_root.exists():
        raise FileNotFoundError(f"Source directory not found: {source_root}")
    if not source_root.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_root}")

    images_by_class: dict[str, list[Path]] = {}
    for image_path in sorted(
        path for path in source_root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    ):
        class_name = infer_class_name(image_path)
        images_by_class.setdefault(class_name, []).append(image_path)

    if not images_by_class:
        raise ValueError(f"No supported image files were found under {source_root}")
    return images_by_class


def validate_ratios(val_ratio: float, test_ratio: float) -> float:
    """Validate split ratios and return the derived train ratio."""
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError(f"test_ratio must be in [0, 1), got {test_ratio}")
    train_ratio = 1.0 - val_ratio - test_ratio
    if train_ratio <= 0.0:
        raise ValueError("train ratio must be positive after subtracting val_ratio and test_ratio")
    return train_ratio


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    """Create an empty output directory."""
    if output_dir.exists():
        if not force:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --force to overwrite it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def resolve_output_dir(data_root: Path, dataset_name: str) -> Path:
    """Resolve the output directory for one dataset."""
    return data_root / "classification" / dataset_name


def allocate_split_counts(total_count: int, split_ratios: dict[str, float]) -> dict[str, int]:
    """Allocate per-class counts while keeping every enabled split non-empty."""
    enabled_splits = [name for name, ratio in split_ratios.items() if ratio > 0.0]
    if total_count < len(enabled_splits):
        raise ValueError(
            "Each class must contain at least "
            f"{len(enabled_splits)} images, got {total_count}."
        )

    counts = {name: 0 for name in split_ratios}
    ratio_sum = sum(split_ratios[name] for name in enabled_splits)
    raw_allocations: dict[str, float] = {}
    for split_name in enabled_splits:
        scaled_value = total_count * (split_ratios[split_name] / ratio_sum)
        raw_allocations[split_name] = scaled_value
        counts[split_name] = math.floor(scaled_value)

    leftover_count = total_count - sum(counts.values())
    ranked_splits = sorted(
        enabled_splits,
        key=lambda name: (raw_allocations[name] - math.floor(raw_allocations[name]), split_ratios[name]),
        reverse=True,
    )
    for split_name in ranked_splits[:leftover_count]:
        counts[split_name] += 1

    empty_splits = [name for name in enabled_splits if counts[name] == 0]
    for split_name in empty_splits:
        donor_name = max(enabled_splits, key=lambda name: counts[name])
        if counts[donor_name] <= 1:
            raise ValueError(
                "Unable to keep every enabled split non-empty for a class with "
                f"{total_count} images."
            )
        counts[donor_name] -= 1
        counts[split_name] += 1

    return counts


def split_one_class(
    image_paths: list[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> dict[str, list[Path]]:
    """Split one class into train/val/test partitions."""
    split_ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }
    counts = allocate_split_counts(len(image_paths), split_ratios)
    shuffled_paths = list(sorted(image_paths))
    rng.shuffle(shuffled_paths)

    train_end = counts["train"]
    val_end = train_end + counts["val"]
    return {
        "train": shuffled_paths[:train_end],
        "val": shuffled_paths[train_end:val_end],
        "test": shuffled_paths[val_end:],
    }


def build_split_mapping(
    images_by_class: dict[str, list[Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> dict[str, dict[str, list[Path]]]:
    """Build split-to-class mappings for the entire dataset."""
    split_mapping = {
        "train": {},
        "val": {},
        "test": {},
    }
    for class_name, image_paths in sorted(images_by_class.items()):
        class_split = split_one_class(
            image_paths=image_paths,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            rng=rng,
        )
        for split_name, split_paths in class_split.items():
            if split_paths:
                split_mapping[split_name][class_name] = split_paths
    return split_mapping


def write_split_manifest(
    split_name: str,
    split_mapping: dict[str, list[Path]],
    source_root: Path,
    output_dir: Path,
) -> int:
    """Write one split manifest and return file count."""
    manifest_path = output_dir / f"{split_name}.txt"
    manifest_lines: list[str] = []
    for class_name, image_paths in sorted(split_mapping.items()):
        for image_path in image_paths:
            relative_path = (Path(source_root.name) / image_path.relative_to(source_root)).as_posix()
            manifest_lines.append(f"{relative_path}\t{class_name}")

    manifest_path.write_text("\n".join(manifest_lines) + ("\n" if manifest_lines else ""), encoding="utf-8")
    return len(manifest_lines)


def print_summary(
    source_root: Path,
    output_dir: Path,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    split_mapping: dict[str, dict[str, list[Path]]],
) -> None:
    """Print a human-readable preparation summary."""
    train_ratio = 1.0 - val_ratio - test_ratio
    print("Prepared NEU-CLS split manifests:")
    print(f"  source_root: {source_root}")
    print(f"  output_dir: {output_dir}")
    print(f"  seed: {seed}")
    print(f"  train_ratio: {train_ratio}")
    print(f"  val_ratio: {val_ratio}")
    print(f"  test_ratio: {test_ratio}")
    for split_name in ("train", "val", "test"):
        file_count = sum(len(paths) for paths in split_mapping[split_name].values())
        print(f"  {split_name}: {file_count} images")
    for class_name in sorted({name for split in split_mapping.values() for name in split}):
        train_count = len(split_mapping["train"].get(class_name, []))
        val_count = len(split_mapping["val"].get(class_name, []))
        test_count = len(split_mapping["test"].get(class_name, []))
        print(
            "  "
            f"{class_name}: train={train_count}, "
            f"val={val_count}, test={test_count}"
        )


def main() -> None:
    """Prepare NEU-CLS split manifests from the flat file layout."""
    args = parse_args()
    source_root = args.source_root.resolve()
    data_root = args.data_root.resolve()
    output_dir = resolve_output_dir(data_root, args.dataset_name).resolve()
    train_ratio = validate_ratios(args.val_ratio, args.test_ratio)
    rng = random.Random(args.seed)

    images_by_class = collect_images_by_class(source_root)
    prepare_output_dir(output_dir, force=args.force)
    split_mapping = build_split_mapping(
        images_by_class=images_by_class,
        train_ratio=train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        rng=rng,
    )

    for split_name, class_mapping in split_mapping.items():
        write_split_manifest(
            split_name=split_name,
            split_mapping=class_mapping,
            source_root=source_root,
            output_dir=output_dir,
        )

    print_summary(
        source_root=source_root,
        output_dir=output_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        split_mapping=split_mapping,
    )


if __name__ == "__main__":
    main()
