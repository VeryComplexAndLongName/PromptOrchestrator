from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel

from .base_client import SummaryLLMClient


class OpenAIConfig(BaseModel):
    api_key: str | None = None
    base_url: str | None = None
    organization: str | None = None


class OpenAISummaryClient(SummaryLLMClient):
    def __init__(self, config: OpenAIConfig | None = None) -> None:
        self.config = config or OpenAIConfig()
        self._client = self._build_client()

    def _build_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI provider requires the 'openai' package. Install dependencies first."
            ) from exc

        return OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            organization=self.config.organization,
        )

    def generate(self, prompt: str, model: str, max_tokens: int, temperature: float) -> str:
        try:
            response = self._client.responses.create(
                model=model,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            text = getattr(response, "output_text", None)
            if text:
                return text
        except Exception:
            pass

        chat = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return chat.choices[0].message.content or ""


def _extract_context_window(payload: object) -> int | None:
    if payload is None:
        return None

    for field in ("context_window", "input_token_limit", "max_input_tokens"):
        value = getattr(payload, field, None)
        if isinstance(value, int) and value > 0:
            return value

    if isinstance(payload, Mapping):
        for field in ("context_window", "input_token_limit", "max_input_tokens"):
            value = payload.get(field)
            if isinstance(value, int) and value > 0:
                return value

    return None


def discover_openai_context_window(config: OpenAIConfig, model: str) -> int | None:
    client = _build_openai_client(config)
    if client is None:
        return None

    try:
        model_payload = client.models.retrieve(model)
        value = _extract_context_window(model_payload)
        if value is not None:
            return value
    except Exception:
        pass

    try:
        models = client.models.list()
        for item in getattr(models, "data", []):
            if getattr(item, "id", None) != model:
                continue
            value = _extract_context_window(item)
            if value is not None:
                return value
    except Exception:
        pass

    return None


def discover_openai_context_window_by_probe(
    config: OpenAIConfig,
    model: str,
    start_size: int = 20000,
    step: int = 2000,
    max_attempts: int = 50,
) -> int | None:
    if start_size <= 0 or step <= 0 or max_attempts <= 0:
        return None

    client = _build_openai_client(config)
    if client is None:
        return None

    attempts_left = max_attempts

    if not _probe_input_size(client=client, model=model, size=start_size):
        return None

    attempts_left -= 1
    best_ok = start_size
    low_ok = start_size
    high_fail: int | None = None

    growth = step
    while attempts_left > 0:
        candidate = low_ok + growth
        if _probe_input_size(client=client, model=model, size=candidate):
            best_ok = candidate
            low_ok = candidate
            growth *= 2
            attempts_left -= 1
            continue

        high_fail = candidate
        attempts_left -= 1
        break

    if high_fail is None:
        return best_ok

    left = low_ok
    right = high_fail - 1

    while attempts_left > 0 and left <= right:
        mid = (left + right) // 2
        if _probe_input_size(client=client, model=model, size=mid):
            best_ok = mid
            left = mid + 1
            attempts_left -= 1
            continue

        right = mid - 1
        attempts_left -= 1

    return best_ok


def _probe_input_size(client: object, model: str, size: int) -> bool:
    try:
        prompt = "A" * size
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
        )
        return True
    except Exception:
        return False


def _build_openai_client(config: OpenAIConfig):
    try:
        from openai import OpenAI
    except ImportError:
        return None

    try:
        return OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            organization=config.organization,
        )
    except Exception:
        return None
