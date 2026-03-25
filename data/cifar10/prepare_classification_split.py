"""Prepare CIFAR-10 into an ImageFolder-style classification split."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from torchvision import datasets


ROOT_DIR = Path(__file__).resolve().parent
RAW_ROOT = ROOT_DIR / "raw"
OUTPUT_ROOT = ROOT_DIR / "classification"


def export_split(split_name: str, train: bool, samples_per_class: int | None) -> int:
    """Export one CIFAR-10 split into ImageFolder layout."""
    dataset = datasets.CIFAR10(root=str(RAW_ROOT), train=train, download=True)
    per_class_counts = {class_name: 0 for class_name in dataset.classes}
    count = 0
    for index, (image, label) in enumerate(dataset):
        class_name = dataset.classes[label]
        if samples_per_class is not None and per_class_counts[class_name] >= samples_per_class:
            continue
        target_dir = OUTPUT_ROOT / split_name / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{split_name}_{index:05d}.png"
        image.save(target_path)
        per_class_counts[class_name] += 1
        count += 1
    return count


def remove_existing_output() -> None:
    """Remove the previous prepared classification directory."""
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Prepare CIFAR-10 into classification/train|val directories.")
    parser.add_argument("--train-per-class", type=int, default=500, help="Number of train images to export per class.")
    parser.add_argument("--val-per-class", type=int, default=100, help="Number of val images to export per class.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Export the full CIFAR-10 dataset instead of a balanced sampled subset.",
    )
    return parser.parse_args()


def main() -> None:
    """Prepare CIFAR-10 classification directories."""
    args = parse_args()
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    remove_existing_output()
    train_limit = None if args.full else args.train_per_class
    val_limit = None if args.full else args.val_per_class
    train_count = export_split("train", train=True, samples_per_class=train_limit)
    val_count = export_split("val", train=False, samples_per_class=val_limit)
    print("Prepared CIFAR-10 classification dataset:")
    print(f"  train: {train_count} images")
    print(f"  val: {val_count} images")
    print(f"Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
