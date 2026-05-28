from __future__ import annotations

from pydantic import BaseModel, Field


class PromptConfig(BaseModel):
    system_prompt: str
    role: str
    task: str
    constraints: list[str] = Field(default_factory=list)
    output_format: str
    examples: list[str] = Field(default_factory=list)

    def render_static_header(self, include_header: bool = False) -> str:
        constraints = "\n".join(f"- {item}" for item in self.constraints) or "- None"
        examples = "\n\n".join(self.examples) or "None"
        header = "=== STATIC PART (CACHE-FRIENDLY) ===\n" if include_header else ""
        return (
            f"{header}"
            "System Prompt:\n"
            f"{self.system_prompt}\n\n"
            "Role:\n"
            f"{self.role}\n\n"
            "Task:\n"
            f"{self.task}\n\n"
            "Constraints:\n"
            f"{constraints}\n\n"
            "Output Format:\n"
            f"{self.output_format}\n\n"
            "Examples:\n"
            f"{examples}"
        )
