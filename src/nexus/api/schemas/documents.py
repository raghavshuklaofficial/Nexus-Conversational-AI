"""Document request/response schemas."""

from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class DocumentUploadRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=1, max_length=500_000)
    source: str = Field(default="", max_length=500)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    document_id: str
    title: str
    status: str
    chunk_count: int = 0
    source: str = ""
    error_message: str | None = None
    created_at: str = ""


class DocumentDeleteResponse(BaseModel):
    status: str = "deleted"
    document_id: str
