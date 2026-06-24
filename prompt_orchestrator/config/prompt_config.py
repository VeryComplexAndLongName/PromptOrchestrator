from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class OutputContractConfig(BaseModel):
    mode: Literal["json_markdown", "json", "markdown", "text"] = "json_markdown"
    strict: bool = True
    schema_hint: str = (
        '{"summary": "str", "findings": ["str"], '
        '"risks": ["str"], "actions": ["str"], "citations": ["str"]}'
    )


class ToolCallingPolicyConfig(BaseModel):
    mode: Literal["allow", "deny", "allowlist"] = "allow"
    max_calls: int = 8
    allowed_tools: list[str] = Field(default_factory=list)
    require_json_arguments: bool = True
    require_tool_result_ack: bool = True


class PromptConfig(BaseModel):
    system_prompt: str
    role: str
    task: str
    constraints: list[str] = Field(default_factory=list)
    output_format: str
    examples: list[str] = Field(default_factory=list)
    response_language: Literal["ru", "en", "auto"] = "ru"
    output_contract: OutputContractConfig = Field(default_factory=OutputContractConfig)
    tool_calling_policy: ToolCallingPolicyConfig = Field(default_factory=ToolCallingPolicyConfig)

    def _render_language_instruction(self) -> str:
        if self.response_language == "ru":
            return "Russian (ru)"
        if self.response_language == "en":
            return "English (en)"
        return "Auto-detect from user query (auto)"

    def _render_output_contract(self) -> str:
        strict = "strict" if self.output_contract.strict else "soft"
        return (
            f"mode={self.output_contract.mode}; "
            f"enforcement={strict}; "
            f"schema={self.output_contract.schema_hint}"
        )

    def _render_tool_policy(self) -> str:
        allowed = ", ".join(self.tool_calling_policy.allowed_tools) or "Any tool"
        return (
            f"mode={self.tool_calling_policy.mode}; "
            f"max_calls={self.tool_calling_policy.max_calls}; "
            f"allowed_tools={allowed}; "
            f"json_args={self.tool_calling_policy.require_json_arguments}; "
            f"ack_result={self.tool_calling_policy.require_tool_result_ack}"
        )

    def render_static_header(self, include_header: bool = False) -> str:
        constraints = "\n".join(f"- {item}" for item in self.constraints) or "- None"
        examples = "\n\n".join(self.examples) or "None"
        response_language = self._render_language_instruction()
        output_contract = self._render_output_contract()
        tool_policy = self._render_tool_policy()
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
            "Response Language:\n"
            f"{response_language}\n\n"
            "Output Format:\n"
            f"{self.output_format}\n\n"
            "Output Contract:\n"
            f"{output_contract}\n\n"
            "Tool Calling Policy:\n"
            f"{tool_policy}\n\n"
            "Examples:\n"
            f"{examples}"
        )
