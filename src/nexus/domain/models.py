"""
Domain models for the Nexus platform.

These are pure data structures with no infrastructure dependencies.
They represent the core business concepts: documents, chunks, citations,
retrieval results, and chat responses.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Document & Chunking
# ---------------------------------------------------------------------------

class DocumentStatus(str, Enum):
    """Lifecycle status of an ingested document."""

    PENDING = "pending"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"
    DELETED = "deleted"


class Document(BaseModel):
    """A document registered for RAG ingestion."""

    id: UUID = Field(default_factory=uuid4)
    title: str = ""
    content: str = ""
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: DocumentStatus = DocumentStatus.PENDING
    chunk_count: int = 0
    error_message: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    idempotency_key: str = ""

    def compute_idempotency_key(self) -> str:
        """Deterministic key based on content hash."""
        h = hashlib.sha256(self.content.encode()).hexdigest()[:16]
        return f"doc:{h}"


class Chunk(BaseModel):
    """A text chunk produced from a document."""

    id: UUID = Field(default_factory=uuid4)
    document_id: str
    content: str
    index: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    token_count: int = 0


# ---------------------------------------------------------------------------
# Retrieval & Citations
# ---------------------------------------------------------------------------

class RetrievedChunk(BaseModel):
    """A chunk returned from vector search with its relevance score."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """A citation linking a response to its source chunk."""

    document_id: str
    chunk_id: str
    score: float
    source: str = ""
    title: str = ""
    snippet: str = ""


# ---------------------------------------------------------------------------
# Chat / Conversation
# ---------------------------------------------------------------------------

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """A single message in a conversation."""

    id: UUID = Field(default_factory=uuid4)
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationMemory(BaseModel):
    """Serialisable conversation history for a session."""

    session_id: str
    messages: list[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        now = datetime.utcnow()
        self.messages.append(Message(role=MessageRole.USER, content=user_text, timestamp=now))
        self.messages.append(Message(role=MessageRole.ASSISTANT, content=assistant_text, timestamp=now))
        self.last_activity = now

    def get_recent_messages(self, n: int = 10) -> list[Message]:
        return self.messages[-n:]

    def format_history(self, n_turns: int = 5) -> str:
        msgs = self.messages[-(n_turns * 2):]
        lines = []
        for m in msgs:
            prefix = "User" if m.role == MessageRole.USER else "Assistant"
            lines.append(f"{prefix}: {m.content}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# RAG Response
# ---------------------------------------------------------------------------

class LatencyBreakdown(BaseModel):
    """Timing breakdown for observability."""

    total_ms: float = 0.0
    nlu_ms: float = 0.0
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    cache_lookup_ms: float = 0.0


class ChatResponse(BaseModel):
    """The final response returned from the chat/RAG pipeline."""

    answer: str
    session_id: str
    intent: str = "unknown"
    entities: list[dict[str, Any]] = Field(default_factory=list)
    sentiment: str = "neutral"
    sentiment_score: float = 0.0
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    cache_hit: bool = False
    suggestions: list[str] = Field(default_factory=list)
    latency: LatencyBreakdown = Field(default_factory=LatencyBreakdown)
    model_name: str = ""
    request_id: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
