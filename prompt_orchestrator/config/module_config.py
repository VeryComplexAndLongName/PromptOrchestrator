from __future__ import annotations

from pydantic import BaseModel, Field

from ..llm.summary_llm import SummaryLLMConfig
from ..safety.llm import SafetyLLMConfig
from .prompt_config import PromptConfig
from .settings import OrchestratorSettings


class ModuleConfig(BaseModel):
    prompt: PromptConfig
    settings: OrchestratorSettings = Field(default_factory=OrchestratorSettings)
    summary_llm: SummaryLLMConfig = Field(default_factory=SummaryLLMConfig)
    safety_llm: SafetyLLMConfig = Field(default_factory=SafetyLLMConfig)
