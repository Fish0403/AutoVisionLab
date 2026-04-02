"""Tests for AIHubMix client retry behavior."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_SITE_PACKAGES = next((REPO_ROOT / ".venv" / "lib").glob("python*/site-packages"))

sys.path.insert(0, str(VENV_SITE_PACKAGES))
sys.path.insert(0, str(REPO_ROOT / "backend"))

import requests

from app.llm.aihubmix_client import (
    AIHUBMIX_REQUEST_TIMEOUT_SECONDS,
    AIHubMixClient,
    AIHubMixRequestError,
)


class AIHubMixClientTest(unittest.TestCase):
    """Verify transient retry handling for provider requests."""

    def _build_settings(self) -> SimpleNamespace:
        return SimpleNamespace(
            aihubmix_api_key="test-key",
            aihubmix_base_url="https://aihubmix.com/v1",
            aihubmix_model="test-model",
        )

    def test_create_json_completion_with_metadata_retries_ssl_eof_then_succeeds(self) -> None:
        success_response = Mock()
        success_response.ok = True
        success_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"ok": true}'
                    }
                }
            ],
            "usage": {"total_tokens": 12},
            "model": "test-model",
        }

        with (
            patch("app.llm.aihubmix_client.get_settings", return_value=self._build_settings()),
            patch(
                "app.llm.aihubmix_client.requests.post",
                side_effect=[
                    requests.exceptions.SSLError("EOF occurred in violation of protocol"),
                    success_response,
                ],
            ) as post_mock,
            patch("app.llm.aihubmix_client.time.sleep") as sleep_mock,
        ):
            client = AIHubMixClient()
            payload, metadata = client.create_json_completion_with_metadata("system", "user")

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(metadata["usage"], {"total_tokens": 12})
        self.assertEqual(post_mock.call_count, 2)
        self.assertEqual(post_mock.call_args.kwargs["headers"]["Connection"], "close")
        self.assertEqual(post_mock.call_args.kwargs["timeout"], AIHUBMIX_REQUEST_TIMEOUT_SECONDS)
        sleep_mock.assert_called_once()

    def test_create_json_completion_with_metadata_retries_retryable_http_status(self) -> None:
        retryable_response = Mock()
        retryable_response.ok = False
        retryable_response.status_code = 503
        retryable_response.text = "upstream unavailable"

        success_response = Mock()
        success_response.ok = True
        success_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"summary_text": "ok"}'
                    }
                }
            ],
            "usage": {"total_tokens": 7},
            "model": "test-model",
        }

        with (
            patch("app.llm.aihubmix_client.get_settings", return_value=self._build_settings()),
            patch(
                "app.llm.aihubmix_client.requests.post",
                side_effect=[retryable_response, success_response],
            ) as post_mock,
            patch("app.llm.aihubmix_client.time.sleep") as sleep_mock,
        ):
            client = AIHubMixClient()
            payload, _ = client.create_json_completion_with_metadata("system", "user")

        self.assertEqual(payload, {"summary_text": "ok"})
        self.assertEqual(post_mock.call_count, 2)
        sleep_mock.assert_called_once()

    def test_create_json_completion_with_metadata_raises_after_retry_budget_is_exhausted(self) -> None:
        with (
            patch("app.llm.aihubmix_client.get_settings", return_value=self._build_settings()),
            patch(
                "app.llm.aihubmix_client.requests.post",
                side_effect=requests.exceptions.SSLError("EOF occurred in violation of protocol"),
            ),
            patch("app.llm.aihubmix_client.time.sleep"),
        ):
            client = AIHubMixClient()
            with self.assertRaisesRegex(AIHubMixRequestError, "request failed after retries"):
                client.create_json_completion_with_metadata("system", "user")


if __name__ == "__main__":
    unittest.main()
