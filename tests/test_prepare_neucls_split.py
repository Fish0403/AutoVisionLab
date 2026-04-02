"""Tests for the flat NEU-CLS split script."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "data" / "prepare_neucls_split.py"


def _write_fake_neucls_tree(root_dir: Path) -> None:
    """Create a minimal flat NEU-CLS-like source tree."""
    root_dir.mkdir(parents=True, exist_ok=True)
    (root_dir / "Cr_1.bmp").write_text("cr-1", encoding="utf-8")
    (root_dir / "Cr_2.bmp").write_text("cr-2", encoding="utf-8")
    (root_dir / "Cr_3.bmp").write_text("cr-3", encoding="utf-8")
    (root_dir / "In_1.bmp").write_text("in-1", encoding="utf-8")
    (root_dir / "In_2.bmp").write_text("in-2", encoding="utf-8")
    (root_dir / "In_3.bmp").write_text("in-3", encoding="utf-8")


class PrepareNeuClsSplitScriptTest(unittest.TestCase):
    """Verify the flat NEU-CLS split script behavior."""

    def test_script_creates_train_and_val_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            source_root = root_dir / "NEU-CLS"
            output_dir = root_dir / "classification" / "NEU"
            _write_fake_neucls_tree(source_root)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--data-root",
                    str(root_dir),
                    "--source-root",
                    str(source_root),
                    "--dataset-name",
                    "NEU",
                    "--val-ratio",
                    "0.5",
                    "--seed",
                    "42",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            train_manifest = output_dir / "train.txt"
            val_manifest = output_dir / "val.txt"
            self.assertTrue(train_manifest.exists())
            self.assertTrue(val_manifest.exists())
            train_lines = [line for line in train_manifest.read_text(encoding="utf-8").splitlines() if line]
            val_lines = [line for line in val_manifest.read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(len(train_lines), 2)
            self.assertEqual(len(val_lines), 2)
            self.assertTrue(all(line.startswith("NEU-CLS/") for line in train_lines))
            self.assertTrue(all(line.startswith("NEU-CLS/") for line in val_lines))
            self.assertIn("Prepared NEU-CLS split manifests:", result.stdout)

    def test_script_requires_force_when_output_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = Path(temp_dir)
            source_root = root_dir / "NEU-CLS"
            output_dir = root_dir / "classification" / "NEU"
            _write_fake_neucls_tree(source_root)
            output_dir.mkdir(parents=True, exist_ok=True)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--data-root",
                    str(root_dir),
                    "--source-root",
                    str(source_root),
                    "--dataset-name",
                    "NEU",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Use --force to overwrite it.", result.stderr)


if __name__ == "__main__":
    unittest.main()
