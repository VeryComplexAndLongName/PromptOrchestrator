from __future__ import annotations

from typing import Protocol


class SummaryLLMClient(Protocol):
    def generate(self, prompt: str, model: str, max_tokens: int, temperature: float) -> str:
        ...
