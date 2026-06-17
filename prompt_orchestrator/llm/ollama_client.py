from __future__ import annotations

import json
import re
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


def discover_ollama_context_window(config: OllamaConfig, model: str) -> int | None:
    endpoint = f"{config.base_url.rstrip('/')}/api/show"
    payload = {"name": model}
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=config.timeout_seconds) as response:
            body = response.read().decode("utf-8")
            parsed = json.loads(body)
    except Exception:
        return None

    if not isinstance(parsed, dict):
        return None

    parameters = parsed.get("parameters")
    if isinstance(parameters, dict):
        value = parameters.get("num_ctx")
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str) and value.strip().isdigit():
            number = int(value.strip())
            return number if number > 0 else None

    if isinstance(parameters, str):
        match = re.search(r"num_ctx\s+(\d+)", parameters)
        if match:
            return int(match.group(1))

    num_ctx = parsed.get("num_ctx")
    if isinstance(num_ctx, int) and num_ctx > 0:
        return num_ctx
    if isinstance(num_ctx, str) and num_ctx.strip().isdigit():
        number = int(num_ctx.strip())
        return number if number > 0 else None

    return None
