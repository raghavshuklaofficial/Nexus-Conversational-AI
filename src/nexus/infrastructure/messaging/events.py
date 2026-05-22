"""
Structured event schemas for the Kafka messaging pipeline.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class BaseEvent(BaseModel):
    """Common envelope for all events."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    idempotency_key: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentIngestRequestedEvent(BaseEvent):
    event_type: str = "document.ingest.requested"
    document_id: str = ""
    title: str = ""
    content: str = ""
    source: str = ""


class DocumentChunkedEvent(BaseEvent):
    event_type: str = "document.chunked"
    document_id: str = ""
    chunk_count: int = 0


class DocumentEmbeddedEvent(BaseEvent):
    event_type: str = "document.embedded"
    document_id: str = ""
    chunk_count: int = 0


class DocumentIndexedEvent(BaseEvent):
    event_type: str = "document.indexed"
    document_id: str = ""
    chunk_count: int = 0
    vector_store: str = "qdrant"


class ChatAnalyticsEvent(BaseEvent):
    event_type: str = "chat.analytics"
    session_id: str = ""
    query: str = ""
    intent: str = ""
    response_length: int = 0
    latency_ms: float = 0.0
    cache_hit: bool = False
    rag_used: bool = False


class DeadLetterEvent(BaseEvent):
    event_type: str = "dead.letter"
    original_event_type: str = ""
    original_event: dict[str, Any] = Field(default_factory=dict)
    error_message: str = ""
    retry_count: int = 0
