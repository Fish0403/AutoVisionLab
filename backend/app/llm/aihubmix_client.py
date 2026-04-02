"""Minimal AIHubMix client."""

from __future__ import annotations

import json
import re
import time
from typing import Any

import requests

from app.core.settings import get_settings


class AIHubMixRequestError(RuntimeError):
    """Raised when AIHubMix returns a request error."""


AIHUBMIX_RETRY_DELAYS_SECONDS = (1, 3)
AIHUBMIX_RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
AIHUBMIX_REQUEST_TIMEOUT_SECONDS = 120


def _build_error_excerpt(text: str | None, *, limit: int = 240) -> str:
    """Return a compact single-line excerpt for provider error reporting."""
    if not text:
        return "<empty>"
    normalized_text = " ".join(text.split())
    if len(normalized_text) <= limit:
        return normalized_text
    return normalized_text[:limit] + "..."


def _iter_json_candidates(content: str) -> list[str]:
    """Return progressively more permissive JSON candidates from one model message."""
    candidates: list[str] = []
    stripped_content = content.strip()
    if stripped_content:
        candidates.append(stripped_content)

    without_think = re.sub(r"<think>.*?</think>", "", stripped_content, flags=re.IGNORECASE | re.DOTALL).strip()
    if without_think and without_think not in candidates:
        candidates.append(without_think)

    without_code_fence = re.sub(r"^```(?:json)?\s*|\s*```$", "", without_think, flags=re.IGNORECASE).strip()
    if without_code_fence and without_code_fence not in candidates:
        candidates.append(without_code_fence)

    first_brace_index = without_code_fence.find("{")
    last_brace_index = without_code_fence.rfind("}")
    if 0 <= first_brace_index < last_brace_index:
        brace_slice = without_code_fence[first_brace_index : last_brace_index + 1].strip()
        if brace_slice and brace_slice not in candidates:
            candidates.append(brace_slice)

    return candidates


def _parse_json_message_content(content: str) -> dict[str, Any]:
    """Parse one provider message into JSON while tolerating common wrapper text."""
    last_error: json.JSONDecodeError | None = None
    for candidate in _iter_json_candidates(content):
        try:
            parsed_candidate = json.loads(candidate)
        except json.JSONDecodeError as error:
            last_error = error
            continue
        if isinstance(parsed_candidate, dict):
            return parsed_candidate
    raise AIHubMixRequestError(
        "chat completions returned non-JSON message content: "
        f"{_build_error_excerpt(content)}"
    ) from last_error


class AIHubMixClient:
    """Thin client for AIHubMix OpenAI-compatible chat completions."""

    def __init__(self) -> None:
        self.settings = get_settings()
        if not self.settings.aihubmix_api_key:
            raise ValueError("AIHUBMIX_API_KEY is not configured")
        self.base_url = self.settings.aihubmix_base_url.strip().rstrip("/")

    def _post_chat_completion(self, payload: dict[str, Any]) -> requests.Response:
        """Post one chat completion request with light retry for transient transport errors."""
        total_attempts = len(AIHUBMIX_RETRY_DELAYS_SECONDS) + 1
        last_error: Exception | None = None
        for attempt_index in range(total_attempts):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.settings.aihubmix_api_key}",
                        "Content-Type": "application/json",
                        "Connection": "close",
                    },
                    json=payload,
                    timeout=AIHUBMIX_REQUEST_TIMEOUT_SECONDS,
                )
            except requests.RequestException as error:
                last_error = error
                if attempt_index >= total_attempts - 1:
                    break
                time.sleep(AIHUBMIX_RETRY_DELAYS_SECONDS[attempt_index])
                continue

            if response.ok:
                return response
            if response.status_code not in AIHUBMIX_RETRYABLE_STATUS_CODES or attempt_index >= total_attempts - 1:
                return response
            last_error = AIHubMixRequestError(
                f"chat completions failed with status={response.status_code} body={response.text}"
            )
            time.sleep(AIHUBMIX_RETRY_DELAYS_SECONDS[attempt_index])

        raise AIHubMixRequestError(
            "chat completions request failed after retries: "
            f"{last_error}"
        ) from last_error

    def create_json_completion_with_metadata(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Request one JSON completion and return parsed content with provider metadata."""
        response = self._post_chat_completion(
            {
                "model": self.settings.aihubmix_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.2,
            }
        )
        if not response.ok:
            raise AIHubMixRequestError(
                f"chat completions failed with status={response.status_code} body={response.text}"
            )
        try:
            payload = response.json()
        except ValueError as error:
            raise AIHubMixRequestError(
                "chat completions returned a non-JSON HTTP body: "
                f"{_build_error_excerpt(response.text)}"
            ) from error

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise AIHubMixRequestError(
                "chat completions payload is missing choices[0].message.content"
            ) from error

        if not isinstance(content, str) or not content.strip():
            raise AIHubMixRequestError("chat completions returned an empty message content")
        parsed_content = _parse_json_message_content(content)
        return (
            parsed_content,
            {
                "usage": payload.get("usage"),
                "response_model": payload.get("model"),
                "response_chars": len(content),
                "raw_content": content,
            },
        )

    def create_json_completion(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Request one JSON completion and parse the response."""
        content, _ = self.create_json_completion_with_metadata(system_prompt, user_prompt)
        return content
