from __future__ import annotations

import re

from ..tokenization import TokenCounter
from .stats import PromptStats


class PromptAnalyzer:
    def __init__(
        self,
        token_model: str = "gpt-4o-mini",
        token_encoding: str | None = None,
    ) -> None:
        self.token_counter = TokenCounter(model=token_model, encoding_name=token_encoding)

    def estimate_tokens(self, text: str) -> int:
        return self.token_counter.count(text)

    def _redundancy_ratio(self, text: str) -> float:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return 0.0
        unique = len(set(lines))
        return max(0.0, 1.0 - (unique / len(lines)))

    def _instruction_density(self, text: str) -> float:
        instruction_markers = [
            r"\bmust\b",
            r"\bshould\b",
            r"\bdo not\b",
            r"\brequired\b",
            r"\bformat\b",
            r"\breturn\b",
        ]
        lower = text.lower()
        hits = sum(len(re.findall(pattern, lower)) for pattern in instruction_markers)
        tokens = max(1, self.estimate_tokens(text))
        return round(hits / tokens, 4)

    def analyze_sections(self, sections: dict[str, str]) -> PromptStats:
        section_chars = {key: len(value) for key, value in sections.items()}
        tokens_by_section = {
            key: self.estimate_tokens(value) for key, value in sections.items()
        }
        total_chars = sum(section_chars.values())
        total_tokens = sum(tokens_by_section.values())
        full_text = "\n\n".join(sections.values())
        redundancy_ratio = self._redundancy_ratio(full_text)
        instruction_density = self._instruction_density(full_text)

        warnings: list[str] = []
        if tokens_by_section.get("rag", 0) > int(total_tokens * 0.4):
            warnings.append("RAG section is too large compared to total prompt.")
        if tokens_by_section.get("recent", 0) > int(total_tokens * 0.35):
            warnings.append("Recent message window dominates the prompt.")
        if redundancy_ratio > 0.45:
            warnings.append("Prompt has high repeated content.")
        if instruction_density < 0.01:
            warnings.append("Instruction density is low; task clarity may suffer.")

        efficiency_score = max(
            0.0,
            min(
                1.0,
                1.0 - (redundancy_ratio * 0.6) - (0.15 if warnings else 0.0),
            ),
        )

        return PromptStats(
            total_chars=total_chars,
            total_tokens=total_tokens,
            section_chars=section_chars,
            tokens_by_section=tokens_by_section,
            warnings=warnings,
            redundancy_ratio=round(redundancy_ratio, 4),
            instruction_density=instruction_density,
            efficiency_score=round(efficiency_score, 4),
        )

    def analyze_prompt(self, prompt: str) -> PromptStats:
        tokens = self.estimate_tokens(prompt)
        redundancy_ratio = self._redundancy_ratio(prompt)
        instruction_density = self._instruction_density(prompt)
        efficiency_score = max(0.0, min(1.0, 1.0 - (redundancy_ratio * 0.6)))
        return PromptStats(
            total_chars=len(prompt),
            total_tokens=tokens,
            section_chars={"full_prompt": len(prompt)},
            tokens_by_section={"full_prompt": tokens},
            warnings=[],
            redundancy_ratio=round(redundancy_ratio, 4),
            instruction_density=instruction_density,
            efficiency_score=round(efficiency_score, 4),
        )
