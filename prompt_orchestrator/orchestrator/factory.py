from __future__ import annotations

from ..cache.base import CacheBackend
from ..cache.local_ttl import LocalTTLCacheBackend
from ..config.config_store import ConfigStore
from ..context.manager import PromptContextManager
from ..llm.base_client import SummaryLLMClient
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
