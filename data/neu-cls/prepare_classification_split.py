"""Prepare NEU-CLS into an ImageFolder-style classification split.

Expected source layout:

data/neu-cls/raw/
  train/train/images/*.jpg
  valid/valid/images/*.jpg

Output layout:

data/neu-cls/classification/
  train/<class_name>/*.jpg
  val/<class_name>/*.jpg
"""

from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SOURCE_SPLITS = {
    "train": ROOT_DIR / "raw" / "train" / "train" / "images",
    "val": ROOT_DIR / "raw" / "valid" / "valid" / "images",
}
OUTPUT_ROOT = ROOT_DIR / "classification"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_VAL_RATIO = 0.2
DEFAULT_SEED = 42


def infer_class_name(image_path: Path) -> str:
    """Infer the class name from the NEU-CLS file name prefix."""
    stem = image_path.stem
    parts = stem.rsplit("_", maxsplit=1)
    if len(parts) != 2 or not parts[1].isdigit():
        raise ValueError(f"Unexpected file name format: {image_path.name}")
    return parts[0]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Prepare NEU-CLS with a stratified train/val split.")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Validation ratio for each class. Default: %(default)s",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for the split. Default: %(default)s",
    )
    return parser.parse_args()


def collect_images_by_class() -> dict[str, list[Path]]:
    """Collect all source images grouped by class name."""
    images_by_class: dict[str, list[Path]] = defaultdict(list)
    for source_dir in SOURCE_SPLITS.values():
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")
        for image_path in sorted(source_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            class_name = infer_class_name(image_path)
            images_by_class[class_name].append(image_path)
    if not images_by_class:
        raise ValueError("No NEU-CLS images were found under the configured source directories")
    return dict(images_by_class)


def reset_output_root() -> None:
    """Recreate the output root to avoid mixing old and new splits."""
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def split_one_class(image_paths: list[Path], val_ratio: float, rng: random.Random) -> tuple[list[Path], list[Path]]:
    """Split one class into train and val partitions."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    shuffled_paths = sorted(image_paths)
    rng.shuffle(shuffled_paths)
    val_count = int(round(len(shuffled_paths) * val_ratio))
    val_count = max(1, min(len(shuffled_paths) - 1, val_count))
    val_paths = shuffled_paths[:val_count]
    train_paths = shuffled_paths[val_count:]
    return train_paths, val_paths


def copy_split(split_name: str, split_mapping: dict[str, list[Path]]) -> int:
    """Copy one split mapping into the output directory and return file count."""
    count = 0
    for class_name, image_paths in sorted(split_mapping.items()):
        target_dir = OUTPUT_ROOT / split_name / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for image_path in image_paths:
            shutil.copy2(image_path, target_dir / image_path.name)
            count += 1
    return count


def main() -> None:
    """Prepare the full NEU-CLS classification directory."""
    args = parse_args()
    rng = random.Random(args.seed)
    images_by_class = collect_images_by_class()
    reset_output_root()

    train_mapping: dict[str, list[Path]] = {}
    val_mapping: dict[str, list[Path]] = {}
    for class_name, image_paths in sorted(images_by_class.items()):
        train_paths, val_paths = split_one_class(image_paths, args.val_ratio, rng)
        train_mapping[class_name] = train_paths
        val_mapping[class_name] = val_paths

    summary = {
        "train": copy_split("train", train_mapping),
        "val": copy_split("val", val_mapping),
    }

    print("Prepared NEU-CLS classification dataset:")
    print(f"  seed: {args.seed}")
    print(f"  val_ratio: {args.val_ratio}")
    for split_name, file_count in summary.items():
        print(f"  {split_name}: {file_count} images")
    for class_name in sorted(images_by_class):
        print(
            "  "
            f"{class_name}: train={len(train_mapping[class_name])}, "
            f"val={len(val_mapping[class_name])}"
        )
    print(f"Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
