"""Unit tests for background experiment suggestion tasks."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_SITE_PACKAGES = next((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"))

sys.path.insert(0, str(VENV_SITE_PACKAGES))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.services import suggestion_service


class SuggestionServiceTest(unittest.TestCase):
    """Verify background suggestion task lifecycle helpers."""

    def setUp(self) -> None:
        suggestion_service.SUGGESTION_TASKS.clear()
        suggestion_service.LATEST_SUGGESTION_TASK_BY_EXPERIMENT.clear()

    def test_start_experiment_suggestion_task_reuses_existing_running_task(self) -> None:
        with (
            patch("app.services.suggestion_service.SessionLocal") as session_local,
            patch(
                "app.services.suggestion_service.get_experiment_detail",
                return_value=SimpleNamespace(id="exp_1", run_id="run_1", status="success"),
            ),
            patch("app.services.suggestion_service.SUGGESTION_EXECUTOR.submit") as submit_mock,
        ):
            session_local.return_value = SimpleNamespace(close=lambda: None)
            first_task = suggestion_service.start_experiment_suggestion_task("exp_1")
            second_task = suggestion_service.start_experiment_suggestion_task("exp_1")

        self.assertEqual(first_task.task_id, second_task.task_id)
        submit_mock.assert_called_once()

    def test_record_provider_usage_updates_task_snapshot(self) -> None:
        suggestion_service.SUGGESTION_TASKS["task_1"] = {
            "task_id": "task_1",
            "experiment_id": "exp_1",
            "run_id": "run_1",
            "status": "running",
            "suggestion": None,
            "error": None,
            "latest_provider_prompt_tokens": None,
            "latest_provider_completion_tokens": None,
            "latest_provider_total_tokens": None,
        }

        suggestion_service._record_provider_usage(
            "task_1",
            {"usage": {"prompt_tokens": 101, "completion_tokens": 29, "total_tokens": 130}},
        )

        task_snapshot = suggestion_service.get_experiment_suggestion_task("exp_1")
        self.assertIsNone(task_snapshot)
        direct_snapshot = suggestion_service._snapshot_suggestion_task("task_1")
        self.assertEqual(direct_snapshot.latest_provider_prompt_tokens, 101)
        self.assertEqual(direct_snapshot.latest_provider_completion_tokens, 29)
        self.assertEqual(direct_snapshot.latest_provider_total_tokens, 130)


if __name__ == "__main__":
    unittest.main()
