"""Unit tests for run-level artifact logging helpers."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "test_run_logging"
VENV_SITE_PACKAGES = next((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"))

os.environ["AVL_ARTIFACT_ROOT"] = str(TEST_ARTIFACT_ROOT)

sys.path.insert(0, str(VENV_SITE_PACKAGES))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.services.run_logging import append_run_prompt_context_event, get_run_prompt_context_log_path


class RunLoggingTest(unittest.TestCase):
    """Verify run-level prompt-context logging behavior."""

    def setUp(self) -> None:
        if TEST_ARTIFACT_ROOT.exists():
            shutil.rmtree(TEST_ARTIFACT_ROOT)

    def tearDown(self) -> None:
        if TEST_ARTIFACT_ROOT.exists():
            shutil.rmtree(TEST_ARTIFACT_ROOT)

    def test_append_run_prompt_context_event_preserves_all_concurrent_appends(self) -> None:
        run_id = "run_concurrent_prompt_context"
        log_path = get_run_prompt_context_log_path(run_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text('{"run_id":"run_concurrent_prompt_context","events":[]}\n', encoding="utf-8")

        original_json_load = json.load

        def delayed_json_load(file_obj):
            time.sleep(0.01)
            return original_json_load(file_obj)

        with patch("app.services.run_logging.json.load", side_effect=delayed_json_load):
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(
                        append_run_prompt_context_event,
                        run_id,
                        "proposal_prompt_context",
                        {"attempt": attempt_index},
                    )
                    for attempt_index in range(20)
                ]
                for future in futures:
                    future.result()

        log_payload = json.loads(log_path.read_text(encoding="utf-8"))
        attempts = sorted(event["payload"]["attempt"] for event in log_payload["events"])

        self.assertEqual(log_payload["run_id"], run_id)
        self.assertEqual(attempts, list(range(20)))
