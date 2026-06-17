from __future__ import annotations

from prompt_orchestrator.llm.openai_client import (
    OpenAIConfig,
    discover_openai_context_window,
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


class _FakeModelMeta:
    def __init__(self, model_id: str) -> None:
        self.id = model_id


class _FakeModelsApi:
    def retrieve(self, model: str):
        _ = model
        return _FakeModelMeta(model_id="without-window")

    def list(self):
        class _List:
            data = [_FakeModelMeta(model_id="qwen3-32b")]

        return _List()


class _FakeOpenAIClientNoWindow:
    def __init__(self) -> None:
        self.models = _FakeModelsApi()


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


def test_discover_openai_context_window_falls_back_to_vllm_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        "prompt_orchestrator.llm.openai_client._build_openai_client",
        lambda config: _FakeOpenAIClientNoWindow(),
    )

    def _fake_http_json(endpoint, method, payload, timeout):
        _ = method
        _ = payload
        _ = timeout
        if endpoint.endswith("/v1/internal/model/info"):
            return {"max_model_len": 32768}
        return None

    monkeypatch.setattr("prompt_orchestrator.llm.openai_client._http_json", _fake_http_json)

    result = discover_openai_context_window(
        config=OpenAIConfig(api_key="x", base_url="http://localhost:8000/v1"),
        model="qwen3-32b",
    )

    assert result == 32768


def test_discover_openai_context_window_falls_back_to_ollama_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        "prompt_orchestrator.llm.openai_client._build_openai_client",
        lambda config: _FakeOpenAIClientNoWindow(),
    )

    def _fake_http_json(endpoint, method, payload, timeout):
        _ = method
        _ = timeout
        if endpoint.endswith("/v1/internal/model/info"):
            return None
        if endpoint.endswith("/api/show") and isinstance(payload, dict) and payload.get("name") == "qwen3-32b":
            return {"parameters": {"num_ctx": 65536}}
        return None

    monkeypatch.setattr("prompt_orchestrator.llm.openai_client._http_json", _fake_http_json)

    result = discover_openai_context_window(
        config=OpenAIConfig(api_key="x", base_url="http://localhost:11434/v1"),
        model="qwen3-32b",
    )

    assert result == 65536
