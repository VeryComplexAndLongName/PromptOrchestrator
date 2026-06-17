from __future__ import annotations

from prompt_orchestrator.llm.openai_client import (
    OpenAIConfig,
    discover_openai_context_window_by_probe,
)


class _FakeCompletions:
    def __init__(self, limit: int) -> None:
        self.limit = limit

    def create(self, model: str, messages: list[dict[str, str]], max_tokens: int) -> None:
        _ = model
        _ = max_tokens
        content = messages[0]["content"]
        if len(content) > self.limit:
            raise RuntimeError("context limit exceeded")


class _FakeChat:
    def __init__(self, limit: int) -> None:
        self.completions = _FakeCompletions(limit=limit)


class _FakeClient:
    def __init__(self, limit: int) -> None:
        self.chat = _FakeChat(limit=limit)


def test_probe_returns_none_on_invalid_params() -> None:
    result = discover_openai_context_window_by_probe(
        config=OpenAIConfig(api_key="x"),
        model="test-model",
        start_size=0,
        step=2000,
        max_attempts=10,
    )

    assert result is None


def test_probe_returns_none_when_first_probe_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        "prompt_orchestrator.llm.openai_client._build_openai_client",
        lambda config: _FakeClient(limit=10000),
    )

    result = discover_openai_context_window_by_probe(
        config=OpenAIConfig(api_key="x"),
        model="test-model",
        start_size=20000,
        step=2000,
        max_attempts=10,
    )

    assert result is None


def test_probe_uses_exponential_and_binary_search(monkeypatch) -> None:
    monkeypatch.setattr(
        "prompt_orchestrator.llm.openai_client._build_openai_client",
        lambda config: _FakeClient(limit=23500),
    )

    result = discover_openai_context_window_by_probe(
        config=OpenAIConfig(api_key="x"),
        model="qwen3-32b",
        start_size=20000,
        step=2000,
        max_attempts=20,
    )

    assert result == 23500


def test_probe_returns_best_known_value_when_attempt_budget_is_exhausted(monkeypatch) -> None:
    monkeypatch.setattr(
        "prompt_orchestrator.llm.openai_client._build_openai_client",
        lambda config: _FakeClient(limit=500000),
    )

    result = discover_openai_context_window_by_probe(
        config=OpenAIConfig(api_key="x"),
        model="qwen3-32b",
        start_size=20000,
        step=2000,
        max_attempts=3,
    )

    assert result == 26000
