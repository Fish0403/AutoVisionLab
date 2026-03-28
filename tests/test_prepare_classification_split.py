"""Tests for the generic class-folder dataset split script."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "data" / "prepare_classification_split.py"


def _write_fake_dataset(source_dir: Path) -> None:
    """Create a small two-class image dataset."""
    samples = {
        "cat": [
            "cat_1.jpg",
            "cat_2.jpg",
            "cat_3.jpg",
            "cat_4.jpg",
            "cat_5.jpg",
        ],
        "dog": [
            "dog_1.jpg",
            "dog_2.jpg",
            "dog_3.jpg",
            "dog_4.jpg",
            "dog_5.jpg",
        ],
    }
    for class_name, file_names in samples.items():
        class_dir = source_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for file_name in file_names:
            (class_dir / file_name).write_text(file_name, encoding="utf-8")


class PrepareClassificationSplitScriptTest(unittest.TestCase):
    """Verify the generic dataset split script behavior."""

    def test_script_creates_train_val_test_manifests_with_dataset_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            dataset_name = "demo-dataset"
            source_dir = root_dir / "raw" / dataset_name
            output_dir = root_dir / "classification" / dataset_name
            _write_fake_dataset(source_dir)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--dataset-name",
                    dataset_name,
                    "--data-root",
                    str(root_dir),
                    "--val-ratio",
                    "0.2",
                    "--test-ratio",
                    "0.2",
                    "--seed",
                    "7",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue((output_dir / "train.txt").exists())
            self.assertTrue((output_dir / "val.txt").exists())
            self.assertTrue((output_dir / "test.txt").exists())
            self.assertFalse((output_dir / "train").exists())
            self.assertEqual(len((output_dir / "train.txt").read_text(encoding="utf-8").strip().splitlines()), 6)
            self.assertEqual(len((output_dir / "val.txt").read_text(encoding="utf-8").strip().splitlines()), 2)
            self.assertEqual(len((output_dir / "test.txt").read_text(encoding="utf-8").strip().splitlines()), 2)
            first_train_line = (output_dir / "train.txt").read_text(encoding="utf-8").splitlines()[0]
            self.assertEqual(len(first_train_line.split("\t")), 2)
            self.assertIn("/", first_train_line.split("\t")[0])
            self.assertIn("Prepared classification dataset:", result.stdout)
            self.assertIn(f"dataset_name: {dataset_name}", result.stdout)

    def test_script_accepts_source_dir_and_infers_dataset_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            source_dir = root_dir / "custom-source"
            output_dir = root_dir / "classification" / "custom-source"
            _write_fake_dataset(source_dir)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--source-dir",
                    str(source_dir),
                    "--data-root",
                    str(root_dir),
                    "--val-ratio",
                    "0.2",
                    "--test-ratio",
                    "0.1",
                    "--seed",
                    "7",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue((output_dir / "train.txt").exists())
            self.assertIn("dataset_name: custom-source", result.stdout)

    def test_script_accepts_source_dir_with_explicit_output_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            source_dir = root_dir / "external-source"
            dataset_name = "kdsc"
            output_dir = root_dir / "classification" / dataset_name
            _write_fake_dataset(source_dir)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--source-dir",
                    str(source_dir),
                    "--dataset-name",
                    dataset_name,
                    "--data-root",
                    str(root_dir),
                    "--val-ratio",
                    "0.2",
                    "--test-ratio",
                    "0.1",
                    "--seed",
                    "7",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertTrue((output_dir / "train.txt").exists())
            self.assertIn(f"dataset_name: {dataset_name}", result.stdout)

    def test_script_requires_force_when_output_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            dataset_name = "demo-dataset"
            source_dir = root_dir / "raw" / dataset_name
            output_dir = root_dir / "classification" / dataset_name
            _write_fake_dataset(source_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--dataset-name",
                    dataset_name,
                    "--data-root",
                    str(root_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Use --force to overwrite it.", result.stderr)


if __name__ == "__main__":
    unittest.main()
