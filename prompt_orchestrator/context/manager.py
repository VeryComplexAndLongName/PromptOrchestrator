from __future__ import annotations

from datetime import datetime, timezone

from ..cache.base import CacheBackend
from ..config.settings import OrchestratorSettings
from ..context.state import DocChunk, Message, PromptContextState
from ..llm.summary_llm import SummaryLLM
from ..tokenization import TokenCounter


class PromptContextManager:
    def __init__(
        self,
        cache_backend: CacheBackend,
        settings: OrchestratorSettings,
        summary_llm: SummaryLLM,
    ) -> None:
        self._cache = cache_backend
        self._settings = settings
        self._summary_llm = summary_llm
        self._token_counter = TokenCounter(
            model=settings.token_model,
            encoding_name=settings.token_encoding,
        )

    def load_state(self, session_id: str) -> PromptContextState:
        cached = self._cache.get(session_id)
        if cached is None:
            return PromptContextState(session_id=session_id)
        return PromptContextState.model_validate(cached)

    def save_state(self, state: PromptContextState) -> None:
        state.last_updated = datetime.now(timezone.utc)
        self._cache.set(
            state.session_id,
            state.model_dump(mode="json"),
            ttl_seconds=self._settings.cache_ttl_seconds,
        )

    def update_state(
        self,
        state: PromptContextState,
        user_message: str,
        model_reply: str | None = None,
    ) -> PromptContextState:
        state.recent_messages.append(Message(role="user", content=user_message))
        if model_reply:
            state.recent_messages.append(Message(role="assistant", content=model_reply))

        state.recent_messages = state.recent_messages[-self._settings.recent_messages_limit :]

        if len(state.recent_messages) >= self._settings.summary_trigger_messages:
            state.summary = self._summary_llm.summarize(
                history=state.recent_messages,
                prev_summary=state.summary,
            )[: self._settings.max_summary_chars]

        self.save_state(state)
        return state

    def set_rag_chunks(self, state: PromptContextState, chunks: list[DocChunk]) -> PromptContextState:
        state.rag_chunks = chunks[: self._settings.rag_limit]
        self.save_state(state)
        return state

    def ensure_fits_limit(self, prompt_payload: dict[str, str]) -> dict[str, str]:
        def payload_text(payload: dict[str, str]) -> str:
            return "\n\n".join(
                [
                    payload.get("static", ""),
                    payload.get("summary", ""),
                    payload.get("recent", ""),
                    payload.get("user", ""),
                    payload.get("rag", ""),
                ]
            )

        def estimate_tokens(text: str) -> int:
            return self._token_counter.count(text)

        while True:
            text = payload_text(prompt_payload)
            if (
                len(text) <= self._settings.max_prompt_chars
                and estimate_tokens(text) <= self._settings.max_prompt_tokens
            ):
                return prompt_payload

            changed = False
            for section in self._settings.section_priority:
                if section == "rag":
                    rag = prompt_payload.get("rag", "")
                    parts = rag.split("\n\n")
                    if len(parts) > 2:
                        prompt_payload["rag"] = "\n\n".join(parts[:-1])
                        changed = True
                        break
                if section == "recent":
                    recent = prompt_payload.get("recent", "")
                    lines = recent.splitlines()
                    if len(lines) > 8:
                        header = lines[:2]
                        body = lines[2:]
                        prompt_payload["recent"] = "\n".join(header + body[2:])
                        changed = True
                        break
                if section == "summary":
                    summary = prompt_payload.get("summary", "")
                    if len(summary) > 200:
                        prompt_payload["summary"] = summary[: int(len(summary) * 0.7)]
                        changed = True
                        break
            if not changed:
                candidates = ["rag", "recent", "summary", "user", "static"]
                largest = max(candidates, key=lambda key: len(prompt_payload.get(key, "")))
                value = prompt_payload.get(largest, "")
                if len(value) <= 40:
                    return prompt_payload
                prompt_payload[largest] = value[: int(len(value) * 0.8)]
