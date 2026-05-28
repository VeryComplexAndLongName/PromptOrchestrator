from __future__ import annotations

import re

from .report import SafetyIssue, SafetyReport


class PromptSafetyEngine:
    INJECTION_PATTERNS = [
        re.compile(r"ignore\s+previous\s+instructions", re.IGNORECASE),
        re.compile(r"(reveal|print|show)\s+(the\s+)?system\s+prompt", re.IGNORECASE),
        re.compile(r"(reveal|print|show)\s+(the\s+)?developer\s+message", re.IGNORECASE),
        re.compile(r"reveal\s+hidden", re.IGNORECASE),
    ]

    def analyze(self, prompt: str) -> SafetyReport:
        issues: list[SafetyIssue] = []

        for pattern in self.INJECTION_PATTERNS:
            if pattern.search(prompt):
                issues.append(
                    SafetyIssue(
                        code="prompt_injection",
                        message=f"Potential injection marker detected: {pattern.pattern}",
                        severity="high",
                    )
                )

        if "always answer in json" in prompt.lower() and "never use json" in prompt.lower():
            issues.append(
                SafetyIssue(
                    code="contradiction",
                    message="Conflicting output constraints found.",
                    severity="medium",
                )
            )

        severity = "none"
        if any(i.severity == "high" for i in issues):
            severity = "high"
        elif any(i.severity == "medium" for i in issues):
            severity = "medium"
        elif issues:
            severity = "low"

        return SafetyReport(issues=issues, severity=severity)

    def sanitize(self, prompt: str) -> str:
        sanitized = prompt
        sanitized = re.sub(
            r"ignore\s+previous\s+instructions",
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
