from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean

from prompt_orchestrator import (
    ConfigStore,
    ModuleConfig,
    OrchestratorSettings,
    PromptConfig,
    PromptOrchestratorFactory,
    SummaryLLMConfig,
)
from prompt_orchestrator.context.state import DocChunk
from prompt_orchestrator.rag.base import RAGProvider


class RagOrchestratorPyPIProvider(RAGProvider):
    def __init__(
        self,
        rag_src: Path,
        db_path: Path,
        table_name: str,
        embed_model: str,
        ollama_url: str,
    ) -> None:
        self.rag_src = rag_src
        self.db_path = db_path

        if not rag_src.exists():
            raise FileNotFoundError(f"rag_orchestrator src not found: {rag_src}")
        if not db_path.exists():
            raise FileNotFoundError(f"RAG database not found: {db_path}")

        if str(rag_src) not in sys.path:
            sys.path.insert(0, str(rag_src))

        from rag_orchestrator.embedding import OllamaEmbedder  # type: ignore
        from rag_orchestrator.factory import create_provider  # type: ignore
        from rag_orchestrator.rag.compat import PromptStyleRAGProviderAdapter  # type: ignore

        provider = create_provider(
            "sqlite+vec",
            db_path=str(db_path),
            table_name=table_name,
        )
        embedder = OllamaEmbedder(
            model=embed_model,
            base_url=ollama_url,
        )
        self._adapter = PromptStyleRAGProviderAdapter(provider=provider, embedder=embedder)

    def retrieve(self, query: str, limit: int) -> list[DocChunk]:
        chunks = self._adapter.retrieve(query=query, limit=limit)
        return [
            DocChunk(
                id=chunk.id,
                content=chunk.content,
                score=chunk.score,
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ]


def build_orchestrator(rag_provider: RAGProvider):
    config = ModuleConfig(
        prompt=PromptConfig(
            system_prompt="You are a Python packages assistant.",
            role="PyPI Support Analyst",
            task="Answer package questions using recent context and RAG snippets.",
            constraints=[
                "Use concise language",
                "Mark uncertain statements",
            ],
            output_format="Markdown",
            examples=["Question: Which package supports data validation?"],
        ),
        settings=OrchestratorSettings(
            max_prompt_chars=3200,
            max_prompt_tokens=900,
            recent_messages_limit=6,
            summary_trigger_messages=4,
            rag_limit=3,
            use_rag_default=True,
        ),
        summary_llm=SummaryLLMConfig(provider="none"),
    )

    return PromptOrchestratorFactory.from_config_store(
        ConfigStore(config),
        rag_provider=rag_provider,
    )


def run_series(use_rag: bool, orchestrator) -> dict[str, float | int]:
    turns = [
        "Что такое pydantic и для чего его обычно используют?",
        "Какие сильные стороны FastAPI по сравнению с другими веб-фреймворками?",
        "Для чего в экосистеме Python обычно применяют httpx?",
    ]

    total_tokens: list[int] = []
    rag_tokens: list[int] = []
    efficiencies: list[float] = []
    warning_count = 0

    mode = "RAG ON" if use_rag else "RAG OFF"
    print(f"\n=== {mode} ===")

    for idx, turn in enumerate(turns, start=1):
        result = orchestrator.build_for_request(
            session_id=f"rag-metrics-{mode.lower().replace(' ', '-')}",
            user_message=turn,
            use_rag=use_rag,
        )
        stats = result.stats

        turn_total_tokens = stats.total_tokens
        turn_rag_tokens = stats.tokens_by_section.get("rag", 0)
        rag_share = (turn_rag_tokens / turn_total_tokens) if turn_total_tokens else 0.0

        total_tokens.append(turn_total_tokens)
        rag_tokens.append(turn_rag_tokens)
        efficiencies.append(stats.efficiency_score)
        warning_count += len(stats.warnings)

        print(
            f"TURN {idx}: total_tokens={turn_total_tokens}, "
            f"rag_tokens={turn_rag_tokens}, rag_share={rag_share:.2%}, "
            f"efficiency={stats.efficiency_score:.4f}"
        )
        if idx == 1:
            rag_preview = result.fitted_sections.get("rag", "")[:220]
            print("rag_section_preview:")
            print(rag_preview if rag_preview else "<empty>")

    return {
        "turns": len(turns),
        "avg_total_tokens": mean(total_tokens),
        "avg_rag_tokens": mean(rag_tokens),
        "avg_rag_share": (sum(rag_tokens) / sum(total_tokens)) if sum(total_tokens) else 0.0,
        "avg_efficiency": mean(efficiencies),
        "warnings": warning_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare prompt metrics with and without RAG")
    parser.add_argument(
        "--rag-src",
        default=r"D:\Prog\AI\RagOrchestrator\src",
        help="Path to rag_orchestrator src directory",
    )
    parser.add_argument(
        "--rag-db",
        default=r"D:\Prog\AI\RagOrchestrator\scripts\pypi_demo\pypi.sqlite",
        help="Path to SQLite+vec PyPI database",
    )
    parser.add_argument(
        "--table-name",
        default="pypi_chunks",
        help="SQLite table name with chunks",
    )
    parser.add_argument(
        "--embed-model",
        default="nomic-embed-text:latest",
        help="Ollama embedding model used for query vectors",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama URL for embedding requests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rag_provider = RagOrchestratorPyPIProvider(
        rag_src=Path(args.rag_src),
        db_path=Path(args.rag_db),
        table_name=args.table_name,
        embed_model=args.embed_model,
        ollama_url=args.ollama_url,
    )
    orchestrator = build_orchestrator(rag_provider)

    metrics_off = run_series(use_rag=False, orchestrator=orchestrator)
    metrics_on = run_series(use_rag=True, orchestrator=orchestrator)

    print("\n=== DELTA (RAG ON - RAG OFF) ===")
    print(f"avg_total_tokens_delta: {metrics_on['avg_total_tokens'] - metrics_off['avg_total_tokens']:.2f}")
    print(f"avg_rag_tokens_delta: {metrics_on['avg_rag_tokens'] - metrics_off['avg_rag_tokens']:.2f}")
    print(f"avg_rag_share_delta: {metrics_on['avg_rag_share'] - metrics_off['avg_rag_share']:.2%}")
    print(f"avg_efficiency_delta: {metrics_on['avg_efficiency'] - metrics_off['avg_efficiency']:.4f}")
    print(f"warnings_delta: {int(metrics_on['warnings']) - int(metrics_off['warnings'])}")


if __name__ == "__main__":
    main()
