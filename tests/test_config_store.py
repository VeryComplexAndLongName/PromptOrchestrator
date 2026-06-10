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
