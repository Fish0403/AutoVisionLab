"""Tests for local dataset discovery helpers."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_SITE_PACKAGES = next((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"))

sys.path.insert(0, str(VENV_SITE_PACKAGES))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.core.settings import Settings
from app.services.dataset_service import build_dataset_summary_text, list_local_datasets
from app.schemas.dataset import LocalDatasetSummary


def _write_image(image_path: Path, size: tuple[int, int]) -> None:
    """Create one RGB image for dataset discovery tests."""
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(1, 2, 3)).save(image_path)


class DatasetServiceTest(unittest.TestCase):
    """Verify local dataset discovery."""

    def test_build_dataset_summary_text_deduplicates_identical_image_sizes(self) -> None:
        summary_text = build_dataset_summary_text(
            LocalDatasetSummary(
                name="NEU",
                is_ready_for_training=True,
                train_sample_count=1260,
                val_sample_count=360,
                test_sample_count=180,
                class_names=["Cr", "In", "PS", "Pa", "RS", "Sc"],
                image_width=200,
                image_height=200,
            ),
            training_image_size=200,
        )

        self.assertEqual(summary_text, "6 classes · train 1260 · val 360 · test 180 · original 200x200")

    def test_list_local_datasets_reports_ready_and_unready_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            ready_dataset_name = f"dataset_{uuid4().hex}"
            missing_manifest_dataset_name = f"dataset_{uuid4().hex}"

            _write_image(data_root / "raw" / ready_dataset_name / "0" / "sample_1.jpg", (64, 64))
            classification_dir = data_root / "classification" / ready_dataset_name
            classification_dir.mkdir(parents=True, exist_ok=True)
            classification_dir.joinpath("train.txt").write_text("0/sample_1.jpg\t0\n", encoding="utf-8")
            classification_dir.joinpath("val.txt").write_text("0/sample_1.jpg\t0\n", encoding="utf-8")

            _write_image(
                data_root / "raw" / "neu" / "classification_source" / "crazing" / "train" / "sample_1.jpg",
                (200, 200),
            )
            (data_root / "raw" / missing_manifest_dataset_name).mkdir(parents=True, exist_ok=True)

            with patch("app.services.dataset_service.get_settings", return_value=Settings(data_root=str(data_root))):
                datasets = list_local_datasets()

            dataset_map = {item.name: item for item in datasets}
            self.assertTrue(dataset_map[ready_dataset_name].is_ready_for_training)
            self.assertEqual(dataset_map[ready_dataset_name].original_image_size, 64)
            self.assertEqual(dataset_map[ready_dataset_name].image_size_options, [64, 96, 128])
            self.assertFalse(dataset_map["neu"].is_ready_for_training)
            self.assertTrue(dataset_map["neu"].has_prepared_source_dir)
            self.assertEqual(dataset_map["neu"].original_image_size, 200)
            self.assertEqual(dataset_map["neu"].image_size_options, [200, 224, 256])
            self.assertFalse(dataset_map[missing_manifest_dataset_name].is_ready_for_training)
            self.assertIn("train.txt is missing", dataset_map[missing_manifest_dataset_name].message or "")

    def test_list_local_datasets_prefers_classification_name_with_case_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = Path(temp_dir)
            raw_dataset_name = f"dataset_{uuid4().hex}"
            classification_dataset_name = raw_dataset_name.upper()
            _write_image(
                data_root / "raw" / raw_dataset_name / "classification_source" / "crazing" / "train" / "sample_1.jpg",
                (200, 200),
            )
            classification_dir = data_root / "classification" / classification_dataset_name
            classification_dir.mkdir(parents=True, exist_ok=True)
            classification_dir.joinpath("train.txt").write_text(
                "crazing/train/sample_1.jpg\tcrazing\n",
                encoding="utf-8",
            )
            classification_dir.joinpath("val.txt").write_text(
                "crazing/train/sample_1.jpg\tcrazing\n",
                encoding="utf-8",
            )

            with patch("app.services.dataset_service.get_settings", return_value=Settings(data_root=str(data_root))):
                datasets = list_local_datasets()

            dataset_map = {item.name: item for item in datasets}
            self.assertIn(classification_dataset_name, dataset_map)
            self.assertTrue(dataset_map[classification_dataset_name].is_ready_for_training)
            self.assertEqual(dataset_map[classification_dataset_name].classification_dir, str(classification_dir))


if __name__ == "__main__":
    unittest.main()
