from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SafetyIssue(BaseModel):
    code: str
    message: str
    severity: Literal["low", "medium", "high"] = "low"


class SafetyReport(BaseModel):
    issues: list[SafetyIssue] = Field(default_factory=list)
    severity: Literal["none", "low", "medium", "high"] = "none"
    sanitized_prompt: str | None = None

    @property
    def is_safe(self) -> bool:
        return self.severity in {"none", "low"}
