from __future__ import annotations

import json
from urllib import request

from pydantic import BaseModel

from .base_client import SummaryLLMClient


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    timeout_seconds: int = 30


class OllamaSummaryClient(SummaryLLMClient):
    def __init__(self, config: OllamaConfig | None = None) -> None:
        self.config = config or OllamaConfig()

    def generate(self, prompt: str, model: str, max_tokens: int, temperature: float) -> str:
        endpoint = f"{self.config.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
            body = response.read().decode("utf-8")
            parsed = json.loads(body)
            return str(parsed.get("response", "")).strip()
