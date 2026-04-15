from __future__ import annotations

import json
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class OpenAICompatibleClient:
    """Minimal OpenAI-compatible Chat Completions client.

    Works with vLLM and LM Studio when they expose an OpenAI-compatible API.
    """

    def __init__(self, base_url: str, api_key: str | None = None, timeout_s: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s

    def chat_completions(self, *, model: str, messages: list[ChatMessage]) -> str:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": 0.2,
        }

        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(url, headers=headers, content=json.dumps(payload))
            resp.raise_for_status()
            data = resp.json()

        # OpenAI-style: choices[0].message.content
        try:
            return str(data["choices"][0]["message"]["content"])
        except Exception as e:  # pragma: no cover
            raise ValueError(f"Unexpected response format from {url}: keys={list(data)}") from e

