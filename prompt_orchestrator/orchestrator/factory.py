from __future__ import annotations

from ..cache.base import CacheBackend
from ..cache.local_ttl import LocalTTLCacheBackend
from ..config.config_store import ConfigStore
from ..context.manager import PromptContextManager
from ..llm.base_client import SummaryLLMClient
from ..llm.openai_client import (
    discover_openai_context_window,
    discover_openai_context_window_by_probe,
)
from ..llm.summary_llm import SummaryLLM
from ..rag.base import RAGProvider
from ..rag.no_rag import NoRAGProvider
from .orchestrator import PromptOrchestrator


class PromptOrchestratorFactory:
    @staticmethod
    def from_config_store(
        config_store: ConfigStore,
        cache_backend: CacheBackend | None = None,
        rag_provider: RAGProvider | None = None,
        summary_llm: SummaryLLM | None = None,
        summary_client: SummaryLLMClient | None = None,
    ) -> PromptOrchestrator:
        settings = config_store.get_settings()
        prompt_config = config_store.get_prompt()
        summary_llm_config = config_store.get_summary_llm()

        if summary_llm_config.provider == "openai":
            discovered_window = discover_openai_context_window(
                config=summary_llm_config.openai,
                model=settings.token_model,
            )
            if discovered_window is None and settings.openai_context_probe_enabled:
                discovered_window = discover_openai_context_window_by_probe(
                    config=summary_llm_config.openai,
                    model=settings.token_model,
                    start_size=settings.openai_context_probe_start_size,
                    step=settings.openai_context_probe_step,
                    max_attempts=settings.openai_context_probe_max_attempts,
                )
            if discovered_window is not None:
                settings.max_prompt_tokens = discovered_window

        cache = cache_backend or LocalTTLCacheBackend(
            default_ttl_seconds=settings.cache_ttl_seconds
        )
        llm = summary_llm or SummaryLLM(
            config=summary_llm_config,
            client=summary_client,
        )
        context_manager = PromptContextManager(
            cache_backend=cache,
            settings=settings,
            summary_llm=llm,
        )
        rag = rag_provider or NoRAGProvider()

        return PromptOrchestrator(
            config=prompt_config,
            context_manager=context_manager,
            rag_provider=rag,
            settings=settings,
            config_store=config_store,
        )
