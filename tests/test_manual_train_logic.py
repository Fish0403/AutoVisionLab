"""Tests for manual train routing decisions in the Streamlit UI."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from frontend.manual_train_logic import resolve_manual_train_action


class ManualTrainLogicTest(unittest.TestCase):
    """Verify manual train requests route to the correct action."""

    def test_all_runs_selection_creates_a_new_run(self) -> None:
        self.assertEqual(
            resolve_manual_train_action(
                selected_run_id="__all__",
                selected_run_model_name=None,
                requested_model_name="mobilenet_v2",
            ),
            "create_run",
        )

    def test_same_model_under_existing_run_appends_experiment(self) -> None:
        self.assertEqual(
            resolve_manual_train_action(
                selected_run_id="run_123",
                selected_run_model_name="mobilenet_v3_small",
                requested_model_name="mobilenet_v3_small",
            ),
            "append_experiment",
        )

    def test_switching_model_under_existing_run_creates_new_run(self) -> None:
        self.assertEqual(
            resolve_manual_train_action(
                selected_run_id="run_123",
                selected_run_model_name="mobilenet_v3_small",
                requested_model_name="mobilenet_v2",
            ),
            "create_run",
        )


if __name__ == "__main__":
    unittest.main()
