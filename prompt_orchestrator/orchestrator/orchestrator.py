from __future__ import annotations

import time

from pydantic import BaseModel

from ..analyzer.analyzer import PromptAnalyzer
from ..analyzer.stats import PromptStats
from ..builder.builder import PromptBuilder
from ..config.config_store import ConfigStore
from ..config.prompt_config import PromptConfig
from ..config.settings import OrchestratorSettings
from ..context.manager import PromptContextManager
from ..context.state import PromptContextState
from ..rag.base import RAGProvider
from ..safety.engine import PromptSafetyEngine
from ..safety.llm import SafetyLLMConfig
from ..safety.report import SafetyReport
from ..telemetry import init_telemetry, telemetry


class OrchestratedPrompt(BaseModel):
    prompt: str
    state: PromptContextState
    stats: PromptStats
    safety: SafetyReport
    sections: dict[str, str]
    fitted_sections: dict[str, str]


class PromptOrchestrator:
    def __init__(
        self,
        config: PromptConfig,
        context_manager: PromptContextManager,
        rag_provider: RAGProvider,
        settings: OrchestratorSettings | None = None,
        config_store: ConfigStore | None = None,
        builder: PromptBuilder | None = None,
        analyzer: PromptAnalyzer | None = None,
        safety_engine: PromptSafetyEngine | None = None,
    ) -> None:
        init_telemetry(service_name="prompt-orchestrator")
        self.config_store = config_store
        self.config = config_store.get_prompt() if config_store else config
        self.context_manager = context_manager
        self.rag_provider = rag_provider
        self.settings = (
            config_store.get_settings()
            if config_store
            else settings or OrchestratorSettings()
        )
        self.builder = builder or PromptBuilder()
        self.analyzer = analyzer or PromptAnalyzer(
            token_model=self.settings.token_model,
            token_encoding=self.settings.token_encoding,
        )
        safety_llm_config = (
            config_store.get_safety_llm()
            if config_store
            else SafetyLLMConfig()
        )
        self.safety_engine = safety_engine or PromptSafetyEngine(llm_config=safety_llm_config)

    def build_for_request(
        self,
        session_id: str,
        user_message: str,
        use_rag: bool | None = None,
    ) -> OrchestratedPrompt:
        started = time.perf_counter()
        with telemetry.span("prompt_orchestrator.build_for_request", {"session.id": session_id}):
            try:
                state = self.context_manager.load_state(session_id)

                if use_rag is None:
                    use_rag = self.settings.use_rag_default
                if use_rag:
                    chunks = self.rag_provider.retrieve(
                        query=user_message,
                        limit=self.settings.rag_limit,
                    )
                    state = self.context_manager.set_rag_chunks(state, chunks)
                else:
                    state = self.context_manager.set_rag_chunks(state, [])

                sections = self.builder.build_sections(
                    config=self.config,
                    state=state,
                    user_message=user_message,
                    include_headers=self.settings.debug_mode,
                )

                fit_payload = self.context_manager.ensure_fits_limit(
                    {
                        "static": sections["static"],
                        "summary": sections["summary"],
                        "recent": sections["recent"],
                        "user": sections["user"],
                        "rag": sections["rag"],
                    }
                )

                prompt = "\n\n".join(
                    [
                        str(fit_payload["static"]),
                        str(fit_payload["summary"]),
                        str(fit_payload["recent"]),
                        str(fit_payload["rag"]),
                    ]
                )

                safety = self.safety_engine.ensure_safe(
                    prompt=prompt,
                    auto_rewrite=self.settings.security_checks_auto_rewrite,
                )
                final_prompt = safety.sanitized_prompt or prompt

                stats = self.analyzer.analyze_sections(
                    {
                        "static": str(fit_payload["static"]),
                        "summary": str(fit_payload["summary"]),
                        "recent": str(fit_payload["recent"]),
                        "rag": str(fit_payload["rag"]),
                    }
                )
                severity_to_score = {"none": 1.0, "low": 0.85, "medium": 0.5, "high": 0.1}
                stats.safety_score = severity_to_score.get(safety.severity, 0.1)

                state = self.context_manager.update_state(state=state, user_message=user_message)

                telemetry.record_build(
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                    total_tokens=stats.total_tokens,
                    total_chars=stats.total_chars,
                    rag_chunks=len(state.rag_chunks),
                    warnings_count=len(stats.warnings),
                    safety_severity=safety.severity,
                    status="ok",
                )

                return OrchestratedPrompt(
                    prompt=final_prompt,
                    state=state,
                    stats=stats,
                    safety=safety,
                    sections=sections,
                    fitted_sections={key: str(value) for key, value in fit_payload.items()},
                )
            except Exception as exc:
                telemetry.record_error("build_for_request", type(exc).__name__)
                telemetry.record_build(
                    duration_ms=(time.perf_counter() - started) * 1000.0,
                    total_tokens=0,
                    total_chars=0,
                    rag_chunks=0,
                    warnings_count=0,
                    safety_severity="unknown",
                    status="error",
                )
                raise
