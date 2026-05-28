from __future__ import annotations

from prompt_orchestrator import (
    ConfigStore,
    ModuleConfig,
    OrchestratorSettings,
    PromptConfig,
    PromptOrchestratorFactory,
    SummaryLLMConfig,
)
from prompt_orchestrator.context.state import Message


def build_orchestrator():
    config = ModuleConfig(
        prompt=PromptConfig(
            system_prompt="You are a context-compaction assistant.",
            role="Conversation Compressor",
            task="Preserve key decisions while keeping prompt size under strict limits.",
            constraints=[
                "Keep factual details only",
                "Drop repeated information",
            ],
            output_format="Markdown",
            examples=["Summarize decisions, open items, and constraints."],
        ),
        settings=OrchestratorSettings(
            max_prompt_chars=950,
            max_prompt_tokens=220,
            recent_messages_limit=8,
            summary_trigger_messages=3,
            max_summary_chars=220,
            rag_limit=0,
            use_rag_default=False,
        ),
        summary_llm=SummaryLLMConfig(provider="none"),
    )
    return PromptOrchestratorFactory.from_config_store(ConfigStore(config))


def main() -> None:
    orchestrator = build_orchestrator()
    session_id = "example-summary-limits"

    turns = [
        "We need a launch checklist for the billing service: SLIs, rollback, and on-call ownership.",
        "Add detailed acceptance criteria for alerts and paging thresholds.",
        "Now include incident communication flow across engineering, support, and product teams.",
        "Add release readiness gates, change freeze windows, and post-release validation steps.",
        "Finally include security sign-off criteria and data retention requirements.",
    ]

    assistant_reply = (
        "Acknowledged. I will preserve decisions, open risks, owners, and deadlines while compacting"
        " repetitive details to keep the context window efficient."
    )

    print("=== SUMMARY + LIMIT FIT EXAMPLE ===")
    for idx, turn in enumerate(turns, start=1):
        result = orchestrator.build_for_request(
            session_id=session_id,
            user_message=turn,
            use_rag=False,
        )
        state = orchestrator.context_manager.load_state(session_id)

        pre_summary_len = len(result.sections.get("summary", ""))
        fitted_summary_len = len(result.fitted_sections.get("summary", ""))
        pre_recent_len = len(result.sections.get("recent", ""))
        fitted_recent_len = len(result.fitted_sections.get("recent", ""))

        summary_present = bool(state.summary)
        summary_truncated = fitted_summary_len < pre_summary_len
        recent_truncated = fitted_recent_len < pre_recent_len

        print(
            f"TURN {idx}: total_tokens={result.stats.total_tokens}, "
            f"summary_present={summary_present}, summary_len={len(state.summary or '')}, "
            f"summary_truncated={summary_truncated}, recent_truncated={recent_truncated}"
        )

        if idx in {3, 5}:
            summary_preview = (state.summary or "")[:180]
            print(f"summary_preview_turn_{idx}: {summary_preview}")

        # Add assistant message to make history richer and trigger stronger compaction pressure.
        state.recent_messages.append(Message(role="assistant", content=assistant_reply))
        state.recent_messages = state.recent_messages[-orchestrator.settings.recent_messages_limit :]
        orchestrator.context_manager.save_state(state)


if __name__ == "__main__":
    main()
