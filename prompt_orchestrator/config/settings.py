from __future__ import annotations

from pydantic import AliasChoices, BaseModel, Field


class OrchestratorSettings(BaseModel):
    max_prompt_chars: int = 16000
    max_prompt_tokens: int = 8000
    token_model: str = "gpt-4o-mini"
    token_encoding: str | None = None
    recent_messages_limit: int = 12
    summary_trigger_messages: int = 20
    cache_ttl_seconds: int = 1800
    rag_limit: int = 5
    use_rag_default: bool = True
    max_summary_chars: int = 2000
    security_checks_auto_rewrite: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "security_checks_auto_rewrite",
            "safety_auto_rewrite",
        ),
    )
    token_chars_ratio: float = 4.0
    openai_context_probe_enabled: bool = False
    openai_context_probe_start_size: int = 20000
    openai_context_probe_step: int = 2000
    openai_context_probe_max_attempts: int = 50

    section_priority: list[str] = Field(
        default_factory=lambda: ["rag", "recent", "summary"]
    )
    debug_mode: bool = False
