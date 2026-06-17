from __future__ import annotations

from prompt_orchestrator import (
    ConfigStore,
    LocalTTLCacheBackend,
    ModuleConfig,
    NoRAGProvider,
    OrchestratorSettings,
    PromptConfig,
    PromptContextManager,
    PromptOrchestrator,
    PromptOrchestratorFactory,
    SafetyLLMConfig,
    SummaryLLM,
    SummaryLLMConfig,
)


def _module_config() -> ModuleConfig:
    return ModuleConfig(
        prompt=PromptConfig(
            system_prompt="You are helpful.",
            role="Architect",
            task="Answer precisely.",
            constraints=["No speculation"],
            output_format="Markdown",
            examples=["Q: A? A: B"],
        ),
        settings=OrchestratorSettings(max_prompt_chars=4000, max_prompt_tokens=1000),
        summary_llm=SummaryLLMConfig(provider="none", model="gpt-4o-mini"),
        safety_llm=SafetyLLMConfig(security_checks_llm_enabled=False),
    )


def test_config_store_gets_values_by_path() -> None:
    store = ConfigStore(_module_config())

    assert store.get("prompt.role") == "Architect"
    assert store.get("settings.max_prompt_tokens") == 1000
    assert store.get("summary_llm.provider") == "none"
    assert store.get("safety_llm.provider") == "ollama"
    assert store.get("missing.path", "default") == "default"


def test_orchestrator_can_use_config_store() -> None:
    store = ConfigStore(_module_config())
    manager = PromptContextManager(
        cache_backend=LocalTTLCacheBackend(),
        settings=store.get_settings(),
        summary_llm=SummaryLLM(config=store.get_summary_llm()),
    )

    orchestrator = PromptOrchestrator(
        config=PromptConfig(
            system_prompt="unused",
            role="unused",
            task="unused",
            constraints=[],
            output_format="text",
            examples=[],
        ),
        context_manager=manager,
        rag_provider=NoRAGProvider(),
        config_store=store,
    )

    result = orchestrator.build_for_request(
        session_id="cfg-session",
        user_message="test config store flow",
        use_rag=False,
    )

    assert "Role:\nArchitect" in result.prompt


def test_factory_builds_orchestrator_from_config_store() -> None:
    store = ConfigStore(_module_config())

    orchestrator = PromptOrchestratorFactory.from_config_store(store)
    result = orchestrator.build_for_request(
        session_id="factory-session",
        user_message="factory flow",
        use_rag=False,
    )

    assert "=== STATIC PART (CACHE-FRIENDLY) ===" not in result.prompt
    assert "Role:\nArchitect" in result.prompt


def test_orchestrator_settings_accepts_legacy_safety_auto_rewrite_alias() -> None:
    settings = OrchestratorSettings.model_validate({"safety_auto_rewrite": False})

    assert settings.security_checks_auto_rewrite is False


def test_factory_replaces_max_prompt_tokens_with_openai_context_window(monkeypatch) -> None:
    cfg = _module_config()
    cfg.summary_llm.provider = "openai"
    cfg.settings.token_model = "gpt-4o-mini"
    cfg.settings.max_prompt_tokens = 1000
    store = ConfigStore(cfg)

    monkeypatch.setattr(
        "prompt_orchestrator.orchestrator.factory.discover_openai_context_window",
        lambda config, model: 128000,
    )

    PromptOrchestratorFactory.from_config_store(
        store,
        summary_llm=SummaryLLM(config=SummaryLLMConfig(provider="none")),
    )

    assert store.get_settings().max_prompt_tokens == 128000


def test_factory_keeps_config_max_prompt_tokens_when_openai_context_window_missing(monkeypatch) -> None:
    cfg = _module_config()
    cfg.summary_llm.provider = "openai"
    cfg.settings.token_model = "gpt-4o-mini"
    cfg.settings.max_prompt_tokens = 1000
    store = ConfigStore(cfg)

    monkeypatch.setattr(
        "prompt_orchestrator.orchestrator.factory.discover_openai_context_window",
        lambda config, model: None,
    )

    PromptOrchestratorFactory.from_config_store(
        store,
        summary_llm=SummaryLLM(config=SummaryLLMConfig(provider="none")),
    )

    assert store.get_settings().max_prompt_tokens == 1000


def test_factory_uses_probe_fallback_when_enabled(monkeypatch) -> None:
    cfg = _module_config()
    cfg.summary_llm.provider = "openai"
    cfg.settings.token_model = "qwen3-32b"
    cfg.settings.max_prompt_tokens = 1000
    cfg.settings.openai_context_probe_enabled = True
    cfg.settings.openai_context_probe_start_size = 20000
    cfg.settings.openai_context_probe_step = 2000
    cfg.settings.openai_context_probe_max_attempts = 10
    store = ConfigStore(cfg)

    monkeypatch.setattr(
        "prompt_orchestrator.orchestrator.factory.discover_openai_context_window",
        lambda config, model: None,
    )
    monkeypatch.setattr(
        "prompt_orchestrator.orchestrator.factory.discover_openai_context_window_by_probe",
        lambda config, model, start_size, step, max_attempts: 36000,
    )

    PromptOrchestratorFactory.from_config_store(
        store,
        summary_llm=SummaryLLM(config=SummaryLLMConfig(provider="none")),
    )

    assert store.get_settings().max_prompt_tokens == 36000


def test_factory_skips_probe_fallback_when_disabled(monkeypatch) -> None:
    cfg = _module_config()
    cfg.summary_llm.provider = "openai"
    cfg.settings.token_model = "qwen3-32b"
    cfg.settings.max_prompt_tokens = 1000
    cfg.settings.openai_context_probe_enabled = False
    store = ConfigStore(cfg)

    monkeypatch.setattr(
        "prompt_orchestrator.orchestrator.factory.discover_openai_context_window",
        lambda config, model: None,
    )
    monkeypatch.setattr(
        "prompt_orchestrator.orchestrator.factory.discover_openai_context_window_by_probe",
        lambda config, model, start_size, step, max_attempts: 36000,
    )

    PromptOrchestratorFactory.from_config_store(
        store,
        summary_llm=SummaryLLM(config=SummaryLLMConfig(provider="none")),
    )

    assert store.get_settings().max_prompt_tokens == 1000
