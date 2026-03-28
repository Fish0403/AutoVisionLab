"""Tests for training failure classification."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_SITE_PACKAGES = next((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"))

sys.path.insert(0, str(VENV_SITE_PACKAGES))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.services.training_runner import build_failure_reason, is_cuda_oom_error


class TrainingRunnerTest(unittest.TestCase):
    """Verify training failure classification."""

    def test_is_cuda_oom_error_detects_oom_message(self) -> None:
        error = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        self.assertTrue(is_cuda_oom_error(error))

    def test_build_failure_reason_formats_oom_hint(self) -> None:
        error = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        failure_reason = build_failure_reason(error)
        self.assertIn("CUDA out of memory.", failure_reason)
        self.assertIn("smaller batch size", failure_reason)
        self.assertIn("Raw error:", failure_reason)


if __name__ == "__main__":
    unittest.main()
