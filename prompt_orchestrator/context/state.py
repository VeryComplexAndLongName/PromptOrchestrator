from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocChunk(BaseModel):
    id: str
    content: str
    score: float | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class PromptContextState(BaseModel):
    session_id: str
    summary: str | None = None
    recent_messages: list[Message] = Field(default_factory=list)
    rag_chunks: list[DocChunk] = Field(default_factory=list)
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
