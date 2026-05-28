from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from prompt_orchestrator import (
    ConfigStore,
    DocChunk,
    ModuleConfig,
    OrchestratorSettings,
    PromptConfig,
    PromptOrchestratorFactory,
    SummaryLLMConfig,
)
from prompt_orchestrator.context.state import Message
from prompt_orchestrator.rag.base import RAGProvider

Turn = tuple[str, str]


class FakeSummaryClient:
    def generate(self, prompt: str, model: str, max_tokens: int, temperature: float) -> str:
        # Deterministic compact summary for repeatable simulation runs.
        tail = prompt[-280:].replace("\n", " ")
        return f"SUMMARY[{model}]: {tail[:220]}"


class FakeRAGProvider(RAGProvider):
    def retrieve(self, query: str, limit: int) -> list[DocChunk]:
        chunks = [
            DocChunk(id="rag-1", content="RAG: Redis is useful for low-latency cache and pub/sub."),
            DocChunk(id="rag-2", content="RAG: Qdrant is a vector database for semantic retrieval."),
            DocChunk(id="rag-3", content="RAG: Keep prompts under token budget via summary and recency window."),
        ]
        return chunks[:limit]


def build_orchestrator(use_rag_default: bool = True, debug_mode: bool = False):
    config = ModuleConfig(
        prompt=PromptConfig(
            system_prompt="You are a prompt orchestrator test assistant.",
            role="Conversation Simulator",
            task="Prepare safe and compact prompt context for each turn.",
            constraints=[
                "Preserve key user facts",
                "Prefer concise context",
                "Do not include unsafe instructions",
            ],
            output_format="Markdown",
            examples=["User asks about cache strategy; assistant keeps summary and recent turns."],
        ),
        settings=OrchestratorSettings(
            max_prompt_chars=2400,
            max_prompt_tokens=420,
            recent_messages_limit=4,
            summary_trigger_messages=4,
            rag_limit=2,
            use_rag_default=use_rag_default,
            debug_mode=debug_mode,
        ),
        summary_llm=SummaryLLMConfig(
            provider="custom",
            model="sim-summary-v1",
            max_tokens=160,
            temperature=0.0,
        ),
    )

    store = ConfigStore(config)
    return PromptOrchestratorFactory.from_config_store(
        config_store=store,
        rag_provider=FakeRAGProvider(),
        summary_client=FakeSummaryClient(),
    )


def log_line(fp, text: str) -> None:
    print(text)
    fp.write(text + "\n")


def load_turns_from_json(path: Path) -> list[Turn]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    turns_raw = payload.get("turns", [])
    turns: list[Turn] = []
    for item in turns_raw:
        user = str(item.get("user", "")).strip()
        assistant = str(item.get("assistant", "")).strip()
        if user:
            turns.append((user, assistant))
    return turns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conversation simulation for prompt_orchestrator")
    parser.add_argument(
        "--turns-file",
        default="test_turns.json",
        help="Path to a JSON file with regular conversation turns.",
    )
    parser.add_argument(
        "--safety-file",
        default="safety_injection_turns.json",
        help="Path to a JSON file with unsafe/injection turns.",
    )
    parser.add_argument(
        "--include-safety",
        action="store_true",
        help="Include turns from safety-file after regular turns.",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG retrieval for this run.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=0,
        help="Limit number of turns to run (0 means all).",
    )
    parser.add_argument(
        "--session-id",
        default="sim-session",
        help="Session id for cache/state isolation.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output with section headers.",
    )
    return parser.parse_args()


def run_simulation() -> None:
    args = parse_args()
    session_id = args.session_id
    use_rag = not args.no_rag
    debug_mode = args.debug
    orchestrator = build_orchestrator(use_rag_default=use_rag, debug_mode=debug_mode)

    turns_file = Path(args.turns_file)
    safety_file = Path(args.safety_file)

    turns = load_turns_from_json(turns_file)
    if args.include_safety:
        turns.extend(load_turns_from_json(safety_file))

    if args.max_turns > 0:
        turns = turns[: args.max_turns]

    log_path = Path(__file__).with_suffix(".log")
    # Overwrite log on each run.
    with log_path.open("w", encoding="utf-8") as log_fp:
        log_line(log_fp, f"Simulation started: {datetime.now().isoformat()}")
        log_line(log_fp, f"Session: {session_id}")
        log_line(log_fp, f"Use RAG: {use_rag}")
        log_line(log_fp, f"Turns file: {turns_file}")
        log_line(log_fp, f"Safety file: {safety_file}")
        log_line(log_fp, f"Include safety: {args.include_safety}")
        log_line(log_fp, f"Turn count: {len(turns)}")
        log_line(log_fp, "=" * 70)

        for idx, (user_msg, assistant_reply) in enumerate(turns, start=1):
            result = orchestrator.build_for_request(
                session_id=session_id,
                user_message=user_msg,
                use_rag=use_rag,
            )

            log_line(log_fp, f"\nTURN {idx}")
            log_line(log_fp, f"USER: {user_msg}")
            log_line(log_fp, "\n=== BUILT PROMPT ===")
            log_line(log_fp, result.prompt)
            log_line(log_fp, "\n=== STATS ===")
            log_line(log_fp, str(result.stats.model_dump()))
            log_line(log_fp, "\n=== SAFETY ===")
            log_line(log_fp, str(result.safety.model_dump()))

            # Simulate assistant reply in history so recent window/summary evolve naturally.
            state = orchestrator.context_manager.load_state(session_id)
            state.recent_messages.append(Message(role="assistant", content=assistant_reply))
            state.recent_messages = state.recent_messages[-orchestrator.settings.recent_messages_limit :]
            orchestrator.context_manager.save_state(state)

            summary_preview = (state.summary or "None")[:220]
            log_line(log_fp, f"\nSUMMARY PREVIEW: {summary_preview}")
            log_line(log_fp, f"RECENT COUNT: {len(state.recent_messages)}")
            log_line(log_fp, "-" * 70)

        log_line(log_fp, f"\nSimulation finished: {datetime.now().isoformat()}")

    print(f"\nLog written to: {log_path}")


if __name__ == "__main__":
    run_simulation()
