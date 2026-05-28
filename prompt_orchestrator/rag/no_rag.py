from __future__ import annotations

from ..context.state import DocChunk
from .base import RAGProvider


class NoRAGProvider(RAGProvider):
    def retrieve(self, query: str, limit: int) -> list[DocChunk]:
        return []
