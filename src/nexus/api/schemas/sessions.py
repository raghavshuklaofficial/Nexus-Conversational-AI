"""Session request/response schemas."""

from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


class SessionResponse(BaseModel):
    session_id: str
    created_at: str
    last_activity: str
    message_count: int = 0


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: list[dict[str, Any]] = Field(default_factory=list)
    total_messages: int = 0


class SessionDeleteResponse(BaseModel):
    status: str = "deleted"
    session_id: str
