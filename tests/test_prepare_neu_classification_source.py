"""Tests for the NEU class-folder preparation script."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "data" / "raw" / "neu" / "prepare_classification_source.py"


def _write_fake_neu_tree(root_dir: Path) -> None:
    """Create a minimal NEU-like source tree."""
    train_dir = root_dir / "train" / "train" / "images"
    valid_dir = root_dir / "valid" / "valid" / "images"
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "crazing_1.jpg").write_text("train-crazing", encoding="utf-8")
    (train_dir / "inclusion_2.jpg").write_text("train-inclusion", encoding="utf-8")
    (valid_dir / "crazing_3.jpg").write_text("valid-crazing", encoding="utf-8")


class PrepareNeuClassificationSourceScriptTest(unittest.TestCase):
    """Verify the NEU preprocessing script behavior."""

    def test_script_creates_class_folders_with_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            source_root = root_dir / "NEU-CLS"
            output_root = root_dir / "classification_source"
            _write_fake_neu_tree(source_root)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--source-root",
                    str(source_root),
                    "--output-root",
                    str(output_root),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            crazing_train = output_root / "crazing" / "train" / "crazing_1.jpg"
            crazing_valid = output_root / "crazing" / "valid" / "crazing_3.jpg"
            inclusion_train = output_root / "inclusion" / "train" / "inclusion_2.jpg"
            self.assertTrue(crazing_train.is_symlink())
            self.assertTrue(crazing_valid.is_symlink())
            self.assertTrue(inclusion_train.is_symlink())
            self.assertIn("Prepared NEU classification source:", result.stdout)

    def test_script_requires_force_when_output_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            source_root = root_dir / "NEU-CLS"
            output_root = root_dir / "classification_source"
            _write_fake_neu_tree(source_root)
            output_root.mkdir(parents=True, exist_ok=True)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--source-root",
                    str(source_root),
                    "--output-root",
                    str(output_root),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Use --force to overwrite it.", result.stderr)


if __name__ == "__main__":
    unittest.main()
