"""Tests for manifest-driven classification data loading."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_SITE_PACKAGES = next((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"))

sys.path.insert(0, str(VENV_SITE_PACKAGES))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.schemas.parameter_space import ExperimentConfig
from app.trainers.classification.base_trainer import BaseClassificationTrainer, ManifestClassificationDataset


def _build_config(dataset_name: str) -> ExperimentConfig:
    """Build a minimal config for data loading tests."""
    return ExperimentConfig.model_validate(
        {
            "task_type": "classification",
            "dataset": dataset_name,
            "model_family": "mobilenet",
            "model_name": "mobilenet_v2",
            "parameter_space_version": "test-v1",
            "params": {
                "optimizer": "adamw",
                "learning_rate": 0.001,
                "batch_size": 2,
                "image_size": 32,
                "epochs": 1,
                "weight_decay": 0.0,
                "scheduler": "none",
                "augmentation_policy": "none",
                "augmentation_params": {
                    "mixup_alpha": 0.0,
                    "cutmix_alpha": 0.0,
                    "random_erasing_prob": 0.0,
                },
                "loss_name": "cross_entropy",
                "loss_params": {"focal_gamma": 2.0},
                "label_smoothing": 0.0,
                "aux_logits": False,
            },
        }
    )


class _DummyTrainer(BaseClassificationTrainer):
    """Concrete trainer used for testing data loading helpers."""

    def create_model(self):
        raise NotImplementedError("Model creation is not needed for dataset loading tests")


def _write_image(image_path: Path) -> None:
    """Create one small RGB test image."""
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color=(12, 34, 56)).save(image_path)


class ClassificationManifestLoaderTest(unittest.TestCase):
    """Verify manifest-backed dataset loading."""

    def test_manifest_dataset_loads_images_from_raw_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            source_root = root_dir / "raw" / "KDSC"
            image_path = source_root / "0" / "sample_1.jpg"
            _write_image(image_path)
            manifest_path = root_dir / "classification" / "KDSC" / "train.txt"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text("0/sample_1.jpg\t0\n", encoding="utf-8")

            dataset = ManifestClassificationDataset(
                manifest_path=manifest_path,
                source_root=source_root,
                class_to_idx={"0": 0},
            )

            image, label = dataset[0]
            self.assertEqual(image.size, (16, 16))
            self.assertEqual(label, 0)

    def test_trainer_resolves_manifest_and_raw_roots_case_insensitively(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            raw_root = root_dir / "raw" / "KDSC"
            classification_root = root_dir / "classification" / "KDSC"
            _write_image(raw_root / "0" / "sample_1.jpg")
            _write_image(raw_root / "1" / "sample_2.jpg")
            classification_root.mkdir(parents=True, exist_ok=True)
            classification_root.joinpath("train.txt").write_text(
                "0/sample_1.jpg\t0\n1/sample_2.jpg\t1\n",
                encoding="utf-8",
            )
            classification_root.joinpath("val.txt").write_text(
                "0/sample_1.jpg\t0\n",
                encoding="utf-8",
            )

            trainer = _DummyTrainer(config=_build_config("kdsc"), experiment_id="exp_1", run_id="run_1")
            trainer.data_root = root_dir
            train_manifest, val_manifest, source_root = trainer.resolve_classification_dataset_files("kdsc")

            self.assertEqual(train_manifest, classification_root / "train.txt")
            self.assertEqual(val_manifest, classification_root / "val.txt")
            self.assertEqual(source_root, raw_root)

    def test_trainer_uses_prepared_source_for_neu(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            raw_root = root_dir / "raw" / "neu"
            prepared_root = raw_root / "classification_source"
            classification_root = root_dir / "classification" / "neu"
            _write_image(prepared_root / "crazing" / "train" / "sample_1.jpg")
            _write_image(prepared_root / "crazing" / "valid" / "sample_2.jpg")
            classification_root.mkdir(parents=True, exist_ok=True)
            classification_root.joinpath("train.txt").write_text(
                "crazing/train/sample_1.jpg\tcrazing\n",
                encoding="utf-8",
            )
            classification_root.joinpath("val.txt").write_text(
                "crazing/valid/sample_2.jpg\tcrazing\n",
                encoding="utf-8",
            )

            trainer = _DummyTrainer(config=_build_config("neu"), experiment_id="exp_1", run_id="run_1")
            trainer.data_root = root_dir
            train_manifest, val_manifest, source_root = trainer.resolve_classification_dataset_files("neu")

            self.assertEqual(train_manifest, classification_root / "train.txt")
            self.assertEqual(val_manifest, classification_root / "val.txt")
            self.assertEqual(source_root, prepared_root)

    def test_demo_subset_indices_mix_multiple_classes(self) -> None:
        trainer = _DummyTrainer(config=_build_config("demo"), experiment_id="exp_1", run_id="run_1")
        dataset = type(
            "DummyDataset",
            (),
            {
                "__len__": lambda self: 9,
                "targets": [0, 0, 0, 0, 1, 1, 1, 2, 2],
            },
        )()

        subset_indices = trainer.build_demo_subset_indices(dataset, max_samples=5, seed=42)
        subset_targets = [dataset.targets[index] for index in subset_indices]

        self.assertEqual(len(subset_indices), 5)
        self.assertEqual(set(subset_targets[:3]), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
