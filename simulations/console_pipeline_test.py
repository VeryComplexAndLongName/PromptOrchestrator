from __future__ import annotations

from pathlib import Path

from prompt_orchestrator import (
    ConfigStore,
    ModuleConfig,
    OllamaConfig,
    OrchestratorSettings,
    PromptConfig,
    PromptOrchestratorFactory,
    SummaryLLMConfig,
)


def build_orchestrator(debug_mode: bool = False):
    config = ModuleConfig(
        prompt=PromptConfig(
            system_prompt="You are a helpful assistant.",
            role="Prompt Pipeline Tester",
            task="Build a safe and concise prompt for each user input.",
            constraints=[
                "Keep answers factual",
                "Avoid unsafe instructions",
            ],
            output_format="Markdown",
            examples=["User: What is TTL cache?"],
        ),
        settings=OrchestratorSettings(
            max_prompt_chars=12000,
            max_prompt_tokens=2500,
            recent_messages_limit=8,
            summary_trigger_messages=4,
            use_rag_default=False,
            debug_mode=debug_mode,
        ),
        summary_llm=SummaryLLMConfig(
            provider="ollama",
            model="codellama:latest",
            ollama=OllamaConfig(
                base_url="http://localhost:11434",
                timeout_seconds=45,
            ),
        ),
    )

    store = ConfigStore(config)
    return PromptOrchestratorFactory.from_config_store(store)


def render_section_body(section_text: str) -> str:
    lines = section_text.splitlines()
    if lines and lines[0].startswith("=== "):
        return "\n".join(lines[1:]).lstrip()
    return section_text


def main() -> None:
    session_id = input("Session ID (default: demo-session): ").strip() or "demo-session"
    debug_mode = input("Enable debug output? [y/N]: ").strip().lower() in {"y", "yes"}
    save_prompts = input("Save full prompts to .txt files? [y/N]: ").strip().lower() in {"y", "yes"}
    
    orchestrator = build_orchestrator(debug_mode=debug_mode)

    print("Type your messages. Type 'exit' to stop.\n")
    while True:
        user_message = input("You> ").strip()
        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit"}:
            print("Stopped.")
            break

        try:
            result = orchestrator.build_for_request(
                session_id=session_id,
                user_message=user_message,
                use_rag=False,
            )
        except Exception as exc:
            print("\n[ERROR] Failed to call Ollama summary model.")
            print("Check that Ollama is running and model 'codellama:latest' is available.")
            print(f"Details: {exc}\n")
            continue

        sections = result.fitted_sections

        print("\n=== BUILT PROMPT ===")
        print(result.prompt)

        if debug_mode:
            print("\n=== STATIC PART (CACHE-FRIENDLY) ===")
            print(render_section_body(sections["static"]))
            print("\n=== SEMI-STABLE PART ===")
            print(render_section_body(sections["summary"]))
            print("\n=== DYNAMIC PART ===")
            print(render_section_body(sections["recent"]))
            print("\n=== MOST DYNAMIC PART (BOTTOM) ===")
            print(render_section_body(sections["rag"]))
            print("\n=== STATS ===")
            print(result.stats.model_dump())
            print("\n=== SAFETY ===")
            print(result.safety.model_dump())

        if save_prompts:
            prompt_dir = Path("saved_prompts")
            prompt_dir.mkdir(exist_ok=True)
            file_name = f"{session_id}_{len(result.state.recent_messages):03d}.txt"
            output_path = prompt_dir / file_name
            output_path.write_text(result.prompt, encoding="utf-8")
            print(f"\nSaved full prompt to: {output_path}")

        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
