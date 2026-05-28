from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import re
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence, Union, get_args, get_origin, get_type_hints

try:
    from pydantic import BaseModel, ConfigDict, Field
except Exception:  # pragma: no cover
    @dataclass
    class _FieldInfo:
        default: Any = None
        default_factory: Optional[Callable[[], Any]] = None

    class BaseModel:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            normalized = self.__class__._normalize(kwargs)
            for key, value in normalized.items():
                setattr(self, key, value)

        @classmethod
        def _coerce(cls, annotation: Any, value: Any) -> Any:
            origin = get_origin(annotation)
            args = get_args(annotation)

            if origin is Union:
                non_none = [item for item in args if item is not type(None)]
                if value is None:
                    return None
                for candidate in non_none:
                    if isinstance(candidate, type) and issubclass(candidate, BaseModel) and isinstance(value, Mapping):
                        return candidate.model_validate(value)
                return value

            if isinstance(annotation, type) and issubclass(annotation, BaseModel) and isinstance(value, Mapping):
                return annotation.model_validate(value)

            return value

        @classmethod
        def _normalize(cls, data: Mapping[str, Any]) -> Dict[str, Any]:
            normalized: Dict[str, Any] = {}
            annotations = get_type_hints(cls)

            for field, annotation in annotations.items():
                if field in data:
                    raw_value = data[field]
                elif hasattr(cls, field):
                    default = getattr(cls, field)
                    if isinstance(default, _FieldInfo):
                        raw_value = default.default_factory() if default.default_factory is not None else default.default
                    else:
                        raw_value = default
                else:
                    continue
                normalized[field] = cls._coerce(annotation, raw_value)

            for key, value in data.items():
                if key not in normalized:
                    normalized[key] = value

            return normalized

        @classmethod
        def model_validate(cls, data: Mapping[str, Any]) -> "BaseModel":
            return cls(**dict(data))

        def model_dump(self) -> Dict[str, Any]:
            return dict(self.__dict__)

    def Field(default: Any = None, default_factory: Optional[Callable[[], Any]] = None) -> Any:
        return _FieldInfo(default=default, default_factory=default_factory)

    ConfigDict = dict  # type: ignore[misc,assignment]

try:
    import tiktoken

    _HAS_TIKTOKEN = True
except Exception:  # pragma: no cover
    tiktoken = None
    _HAS_TIKTOKEN = False


class SummaryProvider(Protocol):
    def summarize(self, text: str, max_tokens: int = 128) -> str:
        ...


class RAGProvider(Protocol):
    def retrieve(self, query: str, top_k: int = 3) -> Sequence[str]:
        ...


class TTLCacheBackend(Protocol):
    def get(self, key: str) -> Optional[str]:
        ...

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        ...


class SummaryConfig(BaseModel):
    provider: str = "extractive"
    max_tokens: int = 128


class CacheConfig(BaseModel):
    enabled: bool = True
    default_ttl_seconds: int = 300


class RagConfig(BaseModel):
    enabled: bool = False
    provider: Optional[str] = None
    top_k: int = 3


class PromptLayoutConfig(BaseModel):
    static_prompt: str = ""
    semi_stable_prompt_template: str = ""


class OrchestratorConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    prompt_layout: PromptLayoutConfig = Field(default_factory=PromptLayoutConfig)
    summary: SummaryConfig = Field(default_factory=SummaryConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    rag: RagConfig = Field(default_factory=RagConfig)


class InMemoryConfigStore:
    def __init__(self, initial: Optional[Mapping[str, Any]] = None) -> None:
        self._data = dict(initial or {})

    def load(self) -> Dict[str, Any]:
        return dict(self._data)

    def save(self, config: Mapping[str, Any]) -> None:
        self._data = dict(config)


class MemoryTTLCache:
    def __init__(self) -> None:
        self._store: Dict[str, tuple[datetime, str]] = {}

    def get(self, key: str) -> Optional[str]:
        payload = self._store.get(key)
        if payload is None:
            return None
        expires_at, value = payload
        if datetime.now(timezone.utc) >= expires_at:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=max(0, ttl_seconds))
        self._store[key] = (expires_at, value)


class NoOpTTLCache:
    def get(self, key: str) -> Optional[str]:
        return None

    def set(self, key: str, value: str, ttl_seconds: int) -> None:
        return None


class ExtractiveSummaryProvider:
    def summarize(self, text: str, max_tokens: int = 128) -> str:
        chunks = [part.strip() for part in re.split(r"[.!?]\s+", text) if part.strip()]
        if not chunks:
            return ""
        summary = chunks[0]
        if len(chunks) > 1 and len(summary.split()) < max_tokens // 2:
            summary = f"{summary}. {chunks[1]}"
        return _truncate_words(summary, max_tokens)


class TruncateSummaryProvider:
    def summarize(self, text: str, max_tokens: int = 128) -> str:
        return _truncate_words(text, max_tokens)


@dataclass
class SafetyReport:
    injection_suspected: bool
    contradiction_suspected: bool
    reasons: list[str]


@dataclass
class EfficiencyReport:
    character_count: int
    word_count: int
    token_count: int
    dynamic_ratio: float


class PromptOrchestrator:
    def __init__(
        self,
        config: OrchestratorConfig,
        summary_provider: SummaryProvider,
        cache_backend: Optional[TTLCacheBackend] = None,
        rag_provider: Optional[RAGProvider] = None,
    ) -> None:
        self.config = config
        self.summary_provider = summary_provider
        self.cache_backend = cache_backend or MemoryTTLCache()
        self.rag_provider = rag_provider

    def build_prompt(
        self,
        dynamic_prompt: str,
        *,
        semi_stable_values: Optional[Mapping[str, Any]] = None,
        include_rag: bool = True,
    ) -> str:
        layout = self.config.prompt_layout
        sections = []
        if layout.static_prompt:
            sections.append(layout.static_prompt.strip())

        if layout.semi_stable_prompt_template:
            values = dict(semi_stable_values or {})
            sections.append(layout.semi_stable_prompt_template.format(**values).strip())

        rag_context = ""
        if include_rag and self.config.rag.enabled and self.rag_provider is not None:
            docs = self.rag_provider.retrieve(dynamic_prompt, top_k=self.config.rag.top_k)
            if docs:
                rag_context = "\n".join(f"- {doc}" for doc in docs)
                sections.append(f"Retrieved context:\n{rag_context}")

        sections.append(dynamic_prompt.strip())
        return "\n\n".join(section for section in sections if section)

    def summarize(self, text: str) -> str:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if self.config.cache.enabled:
            cached = self.cache_backend.get(key)
            if cached is not None:
                return cached

        summary = self.summary_provider.summarize(text, max_tokens=self.config.summary.max_tokens)

        if self.config.cache.enabled:
            self.cache_backend.set(key, summary, self.config.cache.default_ttl_seconds)

        return summary

    def run_safety_checks(self, prompt: str) -> SafetyReport:
        reasons: list[str] = []
        injection_hits = _detect_injection(prompt)
        contradiction_hits = _detect_contradiction(prompt)

        if injection_hits:
            reasons.append("Potential injection phrases detected")
        if contradiction_hits:
            reasons.append("Potential contradictory instructions detected")

        return SafetyReport(
            injection_suspected=bool(injection_hits),
            contradiction_suspected=bool(contradiction_hits),
            reasons=reasons,
        )

    def analyze_efficiency(self, prompt: str) -> EfficiencyReport:
        static_text = self.config.prompt_layout.static_prompt.strip()
        dynamic_words = len(prompt.split())
        total_words = dynamic_words + len(static_text.split())
        dynamic_ratio = 0.0 if total_words == 0 else dynamic_words / total_words

        return EfficiencyReport(
            character_count=len(prompt),
            word_count=dynamic_words,
            token_count=count_tokens(prompt),
            dynamic_ratio=round(dynamic_ratio, 4),
        )


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    if not text.strip():
        return 0

    if _HAS_TIKTOKEN and tiktoken is not None:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except Exception:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    return len(text.split())


def bootstrap_orchestrator(
    config_store: InMemoryConfigStore,
    *,
    summary_providers: Optional[Mapping[str, SummaryProvider]] = None,
    rag_providers: Optional[Mapping[str, RAGProvider]] = None,
    cache_backend: Optional[TTLCacheBackend] = None,
) -> PromptOrchestrator:
    raw_config = config_store.load()
    config = OrchestratorConfig.model_validate(raw_config)

    provider_map: Dict[str, SummaryProvider] = {
        "extractive": ExtractiveSummaryProvider(),
        "truncate": TruncateSummaryProvider(),
    }
    if summary_providers:
        provider_map.update(summary_providers)

    if config.summary.provider not in provider_map:
        raise ValueError(f"Unknown summary provider: {config.summary.provider}")

    rag_provider: Optional[RAGProvider] = None
    if config.rag.enabled and config.rag.provider:
        rag_lookup = dict(rag_providers or {})
        rag_provider = rag_lookup.get(config.rag.provider)
        if rag_provider is None:
            raise ValueError(f"Unknown RAG provider: {config.rag.provider}")

    backend = cache_backend or MemoryTTLCache()

    return PromptOrchestrator(
        config=config,
        summary_provider=provider_map[config.summary.provider],
        cache_backend=backend,
        rag_provider=rag_provider,
    )


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def _detect_injection(text: str) -> bool:
    flags = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"system\s+prompt",
        r"developer\s+message",
        r"jailbreak",
        r"reveal\s+hidden\s+instructions",
    ]
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in flags)


def _detect_contradiction(text: str) -> bool:
    lowered = text.lower()
    contradictory_pairs = [
        ("always", "never"),
        ("must", "must not"),
        ("do", "don't"),
    ]
    return any(left in lowered and right in lowered for left, right in contradictory_pairs)
