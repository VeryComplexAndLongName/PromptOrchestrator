from __future__ import annotations

from pydantic import BaseModel

from ..context.state import DocChunk
from .base import RAGProvider


class QdrantRAGProvider(BaseModel, RAGProvider):
    host: str
    port: int
    collection: str
    client: object

    model_config = {"arbitrary_types_allowed": True}

    def retrieve(self, query: str, limit: int) -> list[DocChunk]:
        result = self.client.query_points(
            collection_name=self.collection,
            query=query,
            limit=limit,
        )
        chunks: list[DocChunk] = []
        for point in getattr(result, "points", []):
            payload = getattr(point, "payload", {}) or {}
            text = payload.get("text") or payload.get("content") or ""
            chunks.append(
                DocChunk(
                    id=str(getattr(point, "id", "")),
                    content=text,
                    score=getattr(point, "score", None),
                    metadata={k: str(v) for k, v in payload.items() if k not in {"text", "content"}},
                )
            )
        return chunks
