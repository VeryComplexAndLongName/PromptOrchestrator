from __future__ import annotations

from prompt_orchestrator import (
    ConfigStore,
    ModuleConfig,
    OrchestratorSettings,
    PromptConfig,
    PromptOrchestratorFactory,
    SummaryLLMConfig,
)


def build_orchestrator():
    config = ModuleConfig(
        prompt=PromptConfig(
            system_prompt="You are a concise technical assistant.",
            role="Senior Python Engineer",
            task="Give practical and short recommendations.",
            constraints=[
                "Do not hallucinate",
                "Explain tradeoffs when they matter",
            ],
            output_format="Markdown",
            examples=["Q: How to speed up tests? A: Use fixtures and isolate IO."],
        ),
        settings=OrchestratorSettings(
            max_prompt_chars=3000,
            max_prompt_tokens=700,
            recent_messages_limit=6,
            summary_trigger_messages=4,
            rag_limit=0,
            use_rag_default=False,
        ),
        summary_llm=SummaryLLMConfig(provider="none"),
    )

    return PromptOrchestratorFactory.from_config_store(ConfigStore(config))


def main() -> None:
    orchestrator = build_orchestrator()
    result = orchestrator.build_for_request(
        session_id="example-basic",
        user_message="Как сократить время выполнения pytest в большом проекте?",
        use_rag=False,
    )

    print("=== PROMPT PREVIEW ===")
    print(result.prompt[:500])

    print("\n=== STATS ===")
    stats = result.stats
    print(f"total_chars: {stats.total_chars}")
    print(f"total_tokens: {stats.total_tokens}")
    print(f"section_chars: {stats.section_chars}")
    print(f"tokens_by_section: {stats.tokens_by_section}")
    print(f"redundancy_ratio: {stats.redundancy_ratio}")
    print(f"instruction_density: {stats.instruction_density}")
    print(f"efficiency_score: {stats.efficiency_score}")
    print(f"safety_score: {stats.safety_score}")
    print(f"warnings: {stats.warnings}")

    print("\n=== SAFETY ===")
    print(result.safety.model_dump())


if __name__ == "__main__":
    main()
