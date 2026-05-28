from __future__ import annotations

from ..config.prompt_config import PromptConfig
from ..context.state import PromptContextState


class PromptBuilder:
    def build_sections(
        self,
        config: PromptConfig,
        state: PromptContextState,
        user_message: str,
        include_headers: bool = False,
    ) -> dict[str, str]:
        static = config.render_static_header(include_header=include_headers)

        summary_header = "=== SEMI-STABLE PART ===\n" if include_headers else ""
        summary = (
            f"{summary_header}"
            "Summary (fixed format):\n"
            f"{state.summary or 'No summary yet.'}"
        )

        recent_lines = [f"{m.role}: {m.content}" for m in state.recent_messages]
        recent_block = "\n".join(recent_lines) or "No recent messages."
        dynamic_header = "=== DYNAMIC PART ===\n" if include_headers else ""
        dynamic = (
            f"{dynamic_header}"
            "Recent Messages (window):\n"
            f"{recent_block}\n\n"
            "User Message:\n"
            f"{user_message}"
        )

        rag_lines = [chunk.content for chunk in state.rag_chunks]
        rag_text = "\n\n".join(rag_lines) or "RAG disabled or no relevant docs."
        rag_header = "=== MOST DYNAMIC PART (BOTTOM) ===\n" if include_headers else ""
        rag = (
            f"{rag_header}"
            "Relevant Docs (RAG):\n"
            f"{rag_text}"
        )

        return {
            "static": static,
            "summary": summary,
            "recent": dynamic,
            "user": user_message,
            "rag": rag,
        }

    def build_prompt(
        self,
        config: PromptConfig,
        state: PromptContextState,
        user_message: str,
        include_headers: bool = False,
    ) -> str:
        sections = self.build_sections(config, state, user_message, include_headers=include_headers)
        return "\n\n".join(
            [sections["static"], sections["summary"], sections["recent"], sections["rag"]]
        )
