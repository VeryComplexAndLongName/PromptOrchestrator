from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

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
        base = prev_summary.strip() + "\n\n" if prev_summary else ""
        transcript = "\n".join(f"{msg.role}: {msg.content}" for msg in history[-30:])

        if self.client is None:
            # Fallback deterministic summarization without external LLM.
            compact = " ".join(line.strip() for line in transcript.splitlines() if line.strip())
            return (base + compact)[:1200]

        prompt = (
            "Summarize the dialogue in a compact, factual format.\n"
            "Keep constraints, decisions, open tasks and user preferences.\n"
            "Avoid speculation and keep under 180 words.\n\n"
            f"Previous summary:\n{prev_summary or 'None'}\n\n"
            f"Dialogue:\n{transcript}"
        )
        return self.client.generate(
            prompt=prompt,
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        ).strip()
