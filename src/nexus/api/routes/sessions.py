"""Session routes — GET/DELETE /sessions"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException

from nexus.api.dependencies import get_session_service
from nexus.api.schemas.sessions import SessionResponse, SessionHistoryResponse, SessionDeleteResponse
from nexus.application.session_service import SessionService

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/sessions", tags=["Sessions"])


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    sessions: SessionService = Depends(get_session_service),
):
    """Get session information."""
    memory = await sessions.get_session(session_id)
    if not memory:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse(
        session_id=memory.session_id,
        created_at=memory.created_at.isoformat(),
        last_activity=memory.last_activity.isoformat(),
        message_count=len(memory.messages),
    )


@router.delete("/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(
    session_id: str,
    sessions: SessionService = Depends(get_session_service),
):
    """Delete a session."""
    deleted = await sessions.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionDeleteResponse(session_id=session_id)
