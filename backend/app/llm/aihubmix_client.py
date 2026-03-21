"""Minimal AIHubMix client."""

from __future__ import annotations

import json

import requests

from app.core.settings import get_settings


class AIHubMixRequestError(RuntimeError):
    """Raised when AIHubMix returns a request error."""


class AIHubMixClient:
    """Thin client for AIHubMix OpenAI-compatible chat completions."""

    def __init__(self) -> None:
        self.settings = get_settings()
        if not self.settings.aihubmix_api_key:
            raise ValueError("AIHUBMIX_API_KEY is not configured")
        self.base_url = self.settings.aihubmix_base_url.strip().rstrip("/")

    def create_json_completion(self, system_prompt: str, user_prompt: str) -> dict:
        """Request one JSON completion and parse the response."""
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.settings.aihubmix_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.settings.aihubmix_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0.2,
            },
            timeout=60,
        )
        if not response.ok:
            raise AIHubMixRequestError(
                f"chat completions failed with status={response.status_code} body={response.text}"
            )
        payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        return json.loads(content)
