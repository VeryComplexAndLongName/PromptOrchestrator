from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from ..telemetry import telemetry
from .base_client import SummaryLLMClient
from .ollama_client import OllamaConfig, OllamaSummaryClient
from .openai_client import OpenAIConfig, OpenAISummaryClient

if TYPE_CHECKING:
    from ..context.state import Message


class SummaryLLMConfig(BaseModel):
    provider: Literal["none", "openai", "ollama", "custom"] = "none"
    model: str = "gpt-4o-mini"
    max_tokens: int = 256
    temperature: float = 0.2
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)


class SummaryLLM:
    def __init__(
        self,
        config: SummaryLLMConfig | None = None,
        client: SummaryLLMClient | None = None,
    ) -> None:
        self.config = config or SummaryLLMConfig()
        self.client = client
        if self.client is None and self.config.provider == "openai":
            self.client = OpenAISummaryClient(config=self.config.openai)
        if self.client is None and self.config.provider == "ollama":
            self.client = OllamaSummaryClient(config=self.config.ollama)

    def summarize(self, history: list[Message], prev_summary: str | None = None) -> str:
        started = time.perf_counter()
        base = prev_summary.strip() + "\n\n" if prev_summary else ""
        transcript = "\n".join(f"{msg.role}: {msg.content}" for msg in history[-30:])

        try:
            if self.client is None:
                # Fallback deterministic summarization without external LLM.
                compact = " ".join(line.strip() for line in transcript.splitlines() if line.strip())
                result = (base + compact)[:1200]
                telemetry.record_summary_call(
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                    provider="none",
                    status="ok",
                )
                return result

            prompt = (
                "Summarize the dialogue in a compact, factual format.\n"
                "Keep constraints, decisions, open tasks and user preferences.\n"
                "Avoid speculation and keep under 180 words.\n\n"
                f"Previous summary:\n{prev_summary or 'None'}\n\n"
                f"Dialogue:\n{transcript}"
            )
            result = self.client.generate(
                prompt=prompt,
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            ).strip()
            telemetry.record_summary_call(
                duration_ms=(time.perf_counter() - started) * 1000.0,
                provider=self.config.provider,
                status="ok",
            )
            return result
        except Exception as exc:
            telemetry.record_error("summary", type(exc).__name__)
            telemetry.record_summary_call(
                duration_ms=(time.perf_counter() - started) * 1000.0,
                provider=self.config.provider,
                status="error",
            )
            raise
