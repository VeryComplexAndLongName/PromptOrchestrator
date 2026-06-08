from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from .report import SafetyIssue, SafetyReport, SafetyThreatGroupReport


Severity = Literal["none", "low", "medium", "high"]


@dataclass(frozen=True, slots=True)
class ThreatRule:
    code: str
    pattern: str | None = None
    contradiction: tuple[str, str] | None = None
    compiled_pattern: re.Pattern[str] | None = None


@dataclass(frozen=True, slots=True)
class ThreatGroup:
    key: str
    description: str
    risk_level: Severity
    weight: float
    rules: tuple[ThreatRule, ...]


def _severity_from_score(score: float) -> Severity:
    if score >= 0.85:
        return "high"
    if score >= 0.5:
        return "medium"
    if score > 0:
        return "low"
    return "none"


def _weight_from_risk_level(risk_level: str) -> float:
    mapping = {
        "high": 1.0,
        "medium": 0.65,
        "low": 0.35,
        "none": 0.0,
    }
    return mapping.get(risk_level, 0.0)


@lru_cache(maxsize=1)
def _load_threat_groups() -> tuple[ThreatGroup, ...]:
    threats_path = Path(__file__).with_name("threats.json")
    raw_data = json.loads(threats_path.read_text(encoding="utf-8"))

    seen_rules: set[tuple[str, str]] = set()
    groups: list[ThreatGroup] = []

    for key, payload in raw_data.items():
        description = str(payload.get("description", ""))
        risk_level = str(payload.get("risk_level", "none"))
        weight = float(payload.get("weight", _weight_from_risk_level(risk_level)))
        rules: list[ThreatRule] = []

        for entry in payload.get("patterns", payload.get("data", [])):
            if "pattern" in entry:
                pattern = str(entry["pattern"])
                dedupe_key = ("pattern", pattern)
                if dedupe_key in seen_rules:
                    continue
                seen_rules.add(dedupe_key)
                rules.append(
                    ThreatRule(
                        code=str(entry["code"]),
                        pattern=pattern,
                        compiled_pattern=re.compile(pattern, re.IGNORECASE),
                    )
                )
        for entry in payload.get("contradictions", []):
            contradiction_values = tuple(str(value) for value in entry["contradiction"])
            dedupe_key = ("contradiction", "||".join(contradiction_values))
            if dedupe_key in seen_rules:
                continue
            seen_rules.add(dedupe_key)
            if len(contradiction_values) != 2:
                continue
            rules.append(
                ThreatRule(
                    code=str(entry["code"]),
                    contradiction=(contradiction_values[0], contradiction_values[1]),
                )
            )

        if not rules and "data" in payload:
            for entry in payload.get("data", []):
                if "pattern" in entry:
                    pattern = str(entry["pattern"])
                    dedupe_key = ("pattern", pattern)
                    if dedupe_key in seen_rules:
                        continue
                    seen_rules.add(dedupe_key)
                    rules.append(
                        ThreatRule(
                            code=str(entry["code"]),
                            pattern=pattern,
                            compiled_pattern=re.compile(pattern, re.IGNORECASE),
                        )
                    )
                elif "contradiction" in entry:
                    contradiction_values = tuple(str(value) for value in entry["contradiction"])
                    dedupe_key = ("contradiction", "||".join(contradiction_values))
                    if dedupe_key in seen_rules:
                        continue
                    seen_rules.add(dedupe_key)
                    if len(contradiction_values) != 2:
                        continue
                    rules.append(
                        ThreatRule(
                            code=str(entry["code"]),
                            contradiction=(contradiction_values[0], contradiction_values[1]),
                        )
                    )

        groups.append(
            ThreatGroup(
                key=key,
                description=description,
                risk_level=risk_level if risk_level in {"none", "low", "medium", "high"} else "none",
                weight=weight,
                rules=tuple(rules),
            )
        )

    return tuple(groups)


class PromptSafetyEngine:
    def __init__(self) -> None:
        self._threat_groups = _load_threat_groups()

    def _too_many_new_lines(self, prompt: str) -> bool:
        return prompt.count("\n") > 300

    def _prompt_too_long(self, prompt: str) -> bool:
        return len(prompt) > 15000

    def analize(self, prompt: str) -> SafetyReport:
        return self.analyze(prompt)

    def analyze(self, prompt: str) -> SafetyReport:
        prompt = unicodedata.normalize("NFKC", prompt)
        prompt_casefold = prompt.casefold()

        issues: list[SafetyIssue] = []
        threat_groups: list[SafetyThreatGroupReport] = []
        highest_score = 0.0

        if self._prompt_too_long(prompt):
            score = 0.5
            issues.append(
                SafetyIssue(
                    code="prompt_too_long",
                    message=f"Prompt length {len(prompt)} exceeds safe threshold.",
                    severity="medium",
                    weight=score,
                )
            )
            highest_score = max(highest_score, score)

        if self._too_many_new_lines(prompt):
            score = 0.5
            issues.append(
                SafetyIssue(
                    code="too_many_newlines",
                    message=f"Prompt contains {prompt.count(chr(10))} newlines, which may indicate an attempt to obfuscate content.",
                    severity="medium",
                    weight=score,
                )
            )
            highest_score = max(highest_score, score)

        for group in self._threat_groups:
            findings: list[SafetyIssue] = []
            for rule in group.rules:
                matched = False
                if rule.compiled_pattern is not None and rule.compiled_pattern.search(prompt):
                    matched = True
                elif rule.contradiction is not None:
                    left = unicodedata.normalize("NFKC", rule.contradiction[0]).casefold()
                    right = unicodedata.normalize("NFKC", rule.contradiction[1]).casefold()
                    matched = left in prompt_casefold and right in prompt_casefold

                if not matched:
                    continue

                severity = _severity_from_score(group.weight)
                issue = SafetyIssue(
                    code=rule.code,
                    message=f"[{group.key}] matched threat code {rule.code}",
                    severity=severity,
                    group=group.key,
                    pattern=rule.pattern if rule.pattern is not None else " / ".join(rule.contradiction or ()),
                    weight=group.weight,
                )
                issues.append(issue)
                findings.append(issue)
                highest_score = max(highest_score, group.weight)

            if findings:
                threat_groups.append(
                    SafetyThreatGroupReport(
                        name=group.key,
                        description=group.description,
                        risk_level=group.risk_level,
                        weight=group.weight,
                        issues=findings,
                    )
                )

        severity = _severity_from_score(highest_score)

        return SafetyReport(
            issues=issues,
            threat_groups=threat_groups,
            severity=severity,
            threat_score=highest_score,
        )

    def sanitize(self, prompt: str) -> str:
        sanitized = prompt
        sanitized = re.sub(
            r"ignore\s+previous\s+instructions",
            "[REMOVED_INJECTION_PATTERN]",
            sanitized,
            flags=re.IGNORECASE,
        )
        sanitized = re.sub(
            r"ignore\s+all\s+rules",
            "[REMOVED_INJECTION_PATTERN]",
            sanitized,
            flags=re.IGNORECASE,
        )
        sanitized = re.sub(
            r"from\s+now\s+on",
            "[REMOVED_INJECTION_PATTERN]",
            sanitized,
            flags=re.IGNORECASE,
        )
        sanitized = re.sub(
            r"reveal\s+hidden",
            "[REMOVED_SENSITIVE_REQUEST]",
            sanitized,
            flags=re.IGNORECASE,
        )
        return sanitized

    def ensure_safe(self, prompt: str, auto_rewrite: bool = True) -> SafetyReport:
        report = self.analyze(prompt)
        if auto_rewrite and report.severity in {"medium", "high"}:
            report.sanitized_prompt = self.sanitize(prompt)
        return report
