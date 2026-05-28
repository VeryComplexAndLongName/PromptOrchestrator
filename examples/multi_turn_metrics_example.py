from __future__ import annotations

from statistics import mean

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
            system_prompt="You are a product analytics assistant.",
            role="Data Analyst",
            task="Build compact context and propose measurable actions.",
            constraints=[
                "Use only provided context",
                "Highlight risks and assumptions",
            ],
            output_format="Markdown",
            examples=["Ask clarifying questions when data is incomplete."],
        ),
        settings=OrchestratorSettings(
            max_prompt_chars=2600,
            max_prompt_tokens=650,
            recent_messages_limit=4,
            summary_trigger_messages=3,
            rag_limit=0,
            use_rag_default=False,
        ),
        summary_llm=SummaryLLMConfig(provider="none"),
    )

    return PromptOrchestratorFactory.from_config_store(ConfigStore(config))


def main() -> None:
    orchestrator = build_orchestrator()

    turns = [
        "Сделай план измерения retention для SaaS-продукта.",
        "Добавь KPI для активации в первую неделю.",
        "Какую структуру A/B теста выбрать для onboarding flow?",
        "Что включить в еженедельный dashboard для PM?",
    ]

    total_tokens = []
    efficiency_scores = []
    safety_scores = []
    warning_count = 0
    severities: dict[str, int] = {"none": 0, "low": 0, "medium": 0, "high": 0}

    for idx, turn in enumerate(turns, start=1):
        result = orchestrator.build_for_request(
            session_id="example-multi-turn",
            user_message=turn,
            use_rag=False,
        )
        stats = result.stats
        safety = result.safety

        total_tokens.append(stats.total_tokens)
        efficiency_scores.append(stats.efficiency_score)
        safety_scores.append(stats.safety_score)
        warning_count += len(stats.warnings)
        severities[safety.severity] += 1

        print(f"TURN {idx}: tokens={stats.total_tokens}, efficiency={stats.efficiency_score}, safety={safety.severity}")

    print("\n=== AGGREGATED KPI ===")
    print(f"turns: {len(turns)}")
    print(f"avg_tokens: {mean(total_tokens):.2f}")
    print(f"max_tokens: {max(total_tokens)}")
    print(f"avg_efficiency_score: {mean(efficiency_scores):.4f}")
    print(f"avg_safety_score: {mean(safety_scores):.4f}")
    print(f"total_warnings: {warning_count}")
    print(f"safety_severity_distribution: {severities}")


if __name__ == "__main__":
    main()
