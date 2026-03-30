"""Tests for dataset-aware classification augmentations."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

from torchvision.transforms import Normalize


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_SITE_PACKAGES = next((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"))

sys.path.insert(0, str(VENV_SITE_PACKAGES))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.schemas.parameter_space import TrainHypAugmentation
from app.trainers.classification.components.augmentations import (
    build_eval_transform,
    build_train_transform,
    resolve_classification_normalization,
)


class ClassificationAugmentationsTest(unittest.TestCase):
    """Verify dataset-aware normalization behavior."""

    def test_resolve_classification_normalization_returns_dataset_stats(self) -> None:
        mean, std = resolve_classification_normalization("DT")  # type: ignore[misc]

        self.assertEqual(mean, (0.4707, 0.4707, 0.4707))
        self.assertEqual(std, (0.0587, 0.0587, 0.0587))

    def test_build_train_transform_uses_dataset_specific_normalization(self) -> None:
        augmentation = TrainHypAugmentation(policy="basic", mixup=0.0, cutmix=0.0, random_erasing=0.0)

        transform = build_train_transform(224, augmentation, dataset_name="NEU")

        self.assertIsInstance(transform.transforms[-1], Normalize)
        self.assertEqual(transform.transforms[-1].mean, (0.5002, 0.5002, 0.5002))
        self.assertEqual(transform.transforms[-1].std, (0.1103, 0.1103, 0.1103))

    def test_build_eval_transform_skips_unknown_dataset_normalization(self) -> None:
        transform = build_eval_transform(224, dataset_name="unknown-dataset")

        self.assertFalse(any(isinstance(step, Normalize) for step in transform.transforms))


if __name__ == "__main__":
    unittest.main()
