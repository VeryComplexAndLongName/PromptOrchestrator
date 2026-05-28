from __future__ import annotations

from pydantic import BaseModel, Field


class PromptStats(BaseModel):
    total_chars: int
    total_tokens: int
    section_chars: dict[str, int] = Field(default_factory=dict)
    tokens_by_section: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    redundancy_ratio: float = 0.0
    instruction_density: float = 0.0
    efficiency_score: float = 1.0
    safety_score: float = 1.0
