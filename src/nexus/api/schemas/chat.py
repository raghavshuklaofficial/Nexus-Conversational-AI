"""Chat request/response schemas."""

from __future__ import annotations
from typing import Any
from uuid import UUID
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096, description="User message")
    session_id: str | None = Field(default=None, description="Session ID for context")
    use_rag: bool = Field(default=False, description="Enable RAG retrieval")
    stream: bool = Field(default=False, description="Enable streaming")
    top_k: int = Field(default=5, ge=1, le=20, description="Chunks to retrieve")

    class Config:
        json_schema_extra = {
            "example": {
                "text": "What is retrieval-augmented generation?",
                "use_rag": True,
                "top_k": 5,
            }
        }


class CitationSchema(BaseModel):
    document_id: str
    chunk_id: str
    score: float
    source: str = ""
    title: str = ""
    snippet: str = ""


class RetrievedChunkSchema(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    score: float


class LatencySchema(BaseModel):
    total_ms: float = 0.0
    nlu_ms: float = 0.0
    retrieval_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    cache_lookup_ms: float = 0.0


class ChatResponseSchema(BaseModel):
    answer: str
    session_id: str
    intent: str = "unknown"
    entities: list[dict[str, Any]] = Field(default_factory=list)
    sentiment: str = "neutral"
    sentiment_score: float = 0.0
    citations: list[CitationSchema] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunkSchema] = Field(default_factory=list)
    cache_hit: bool = False
    suggestions: list[str] = Field(default_factory=list)
    latency: LatencySchema = Field(default_factory=LatencySchema)
    model_name: str = ""
    request_id: str = ""
