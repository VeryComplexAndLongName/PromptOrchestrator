from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Severity = Literal["none", "low", "medium", "high"]


class SafetyIssue(BaseModel):
    code: str
    message: str
    severity: Severity = "none"
    group: str | None = None
    pattern: str | None = None
    weight: float | None = None


class SafetyThreatGroupReport(BaseModel):
    name: str
    description: str
    risk_level: Severity = "none"
    weight: float = 0.0
    issues: list[SafetyIssue] = Field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.issues)

    @property
    def codes(self) -> list[str]:
        return [issue.code for issue in self.issues]


class SafetyReport(BaseModel):
    issues: list[SafetyIssue] = Field(default_factory=list)
    threat_groups: list[SafetyThreatGroupReport] = Field(default_factory=list)
    severity: Severity = "none"
    threat_score: float = 0.0
    sanitized_prompt: str | None = None

    @property
    def is_safe(self) -> bool:
        return self.severity in {"none", "low"}

    @property
    def grouped_summary(self) -> str:
        if not self.threat_groups:
            return ""

        lines: list[str] = []
        for index, group in enumerate(self.threat_groups, start=1):
            codes = ", ".join(group.codes) if group.codes else "None"
            lines.append(f"{index}. {group.name}: {group.count} threat(s), codes: {codes}")
        return "\n".join(lines)
