"""
API Routes
==========

REST API endpoints for the conversational AI system.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from nexus.core.engine import ConversationEngine
from nexus.core.session import ConversationSession

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["Conversation"])


# Request/Response Models

class MessageRequest(BaseModel):
    """Request model for sending a message."""
    
    text: str = Field(..., min_length=1, max_length=4096, description="User message text")
    session_id: UUID | None = Field(default=None, description="Optional session ID for context")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Hello, how can you help me?",
                "session_id": None,
            }
        }


class MessageResponse(BaseModel):
    """Response model for messages."""
    
    id: str = Field(..., description="Response ID")
    text: str = Field(..., description="Response text")
    type: str = Field(..., description="Response type")
    session_id: str = Field(..., description="Session ID")
    suggestions: list[str] = Field(default_factory=list, description="Follow-up suggestions")
    sentiment: str = Field(..., description="Detected sentiment")
    confidence: float = Field(..., ge=0, le=1, description="Intent confidence")
    intent: str = Field(..., description="Detected intent")
    entities: list[dict[str, Any]] = Field(default_factory=list, description="Extracted entities")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    timestamp: str = Field(..., description="Response timestamp")


class SessionResponse(BaseModel):
    """Response model for session information."""
    
    session_id: str
    created_at: str
    turn_count: int
    is_expired: bool


class SessionHistoryResponse(BaseModel):
    """Response model for session history."""
    
    session_id: str
    turns: list[dict[str, Any]]
    total_turns: int


class AnalyzeRequest(BaseModel):
    """Request model for text analysis."""
    
    text: str = Field(..., min_length=1, max_length=4096)


class AnalyzeResponse(BaseModel):
    """Response model for text analysis."""
    
    intent: dict[str, Any]
    entities: list[dict[str, Any]]
    sentiment: dict[str, Any]
    processing_time_ms: float


# Dependencies

def get_engine(request: Request) -> ConversationEngine:
    """Get the conversation engine from app state."""
    if not hasattr(request.app.state, "engine"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conversation engine not initialized",
        )
    return request.app.state.engine


# Endpoints

@router.post("/chat", response_model=MessageResponse)
async def chat(
    request: MessageRequest,
    engine: ConversationEngine = Depends(get_engine),
) -> MessageResponse:
    """
    Send a message and receive a response.
    
    This is the main conversation endpoint. Optionally provide a session_id
    for multi-turn conversations with context.
    """
    try:
        # Get or create session
        session: ConversationSession | None = None
        
        if request.session_id:
            session = await engine.get_session(request.session_id)
            if not session:
                logger.warning("session_not_found", session_id=str(request.session_id))
        
        if not session:
            session = await engine.create_session()
        
        # Process message
        response = await engine.process(request.text, session=session)
        
        # Build response
        return MessageResponse(
            id=str(response.id),
            text=response.text,
            type=response.type.value,
            session_id=str(session.id),
            suggestions=response.suggestions,
            sentiment=response.metadata.sentiment.value,
            confidence=response.metadata.detected_intent.confidence if response.metadata.detected_intent else 0.0,
            intent=response.metadata.detected_intent.name if response.metadata.detected_intent else "unknown",
            entities=[
                {
                    "text": e.text,
                    "type": e.type,
                    "value": e.value,
                    "confidence": e.confidence,
                }
                for e in response.metadata.extracted_entities
            ],
            processing_time_ms=response.metadata.processing_time_ms,
            timestamp=response.created_at.isoformat(),
        )
        
    except Exception as e:
        logger.error("chat_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}",
        )


@router.post("/sessions", response_model=SessionResponse)
async def create_session(
    engine: ConversationEngine = Depends(get_engine),
) -> SessionResponse:
    """Create a new conversation session."""
    session = await engine.create_session()
    
    return SessionResponse(
        session_id=str(session.id),
        created_at=session.created_at.isoformat(),
        turn_count=session.turn_count,
        is_expired=session.is_expired,
    )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: UUID,
    engine: ConversationEngine = Depends(get_engine),
) -> SessionResponse:
    """Get information about a specific session."""
    session = await engine.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
    
    return SessionResponse(
        session_id=str(session.id),
        created_at=session.created_at.isoformat(),
        turn_count=session.turn_count,
        is_expired=session.is_expired,
    )


@router.get("/sessions/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(
    session_id: UUID,
    engine: ConversationEngine = Depends(get_engine),
) -> SessionHistoryResponse:
    """Get conversation history for a session."""
    session = await engine.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
    
    return SessionHistoryResponse(
        session_id=str(session.id),
        turns=[
            {
                "user_input": turn.user_input,
                "response_text": turn.response.text,
                "response_type": turn.response.type.value,
                "timestamp": turn.timestamp.isoformat(),
            }
            for turn in session.history
        ],
        total_turns=session.turn_count,
    )


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: UUID,
    engine: ConversationEngine = Depends(get_engine),
) -> dict:
    """Delete a conversation session."""
    session = await engine.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )
    
    # Remove from engine's sessions
    if session_id in engine._sessions:
        del engine._sessions[session_id]
    
    return {"status": "deleted", "session_id": str(session_id)}


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(
    request: AnalyzeRequest,
    engine: ConversationEngine = Depends(get_engine),
) -> AnalyzeResponse:
    """
    Analyze text without generating a response.
    
    Returns intent classification, entity extraction, and sentiment analysis results.
    """
    import time
    
    start = time.time()
    
    try:
        # Run NLU pipeline
        import asyncio
        
        intent_result, entities, sentiment_result = await asyncio.gather(
            engine._intent_classifier.classify(request.text),
            engine._entity_extractor.extract(request.text),
            engine._sentiment_analyzer.analyze(request.text),
        )
        
        processing_time = (time.time() - start) * 1000
        
        return AnalyzeResponse(
            intent={
                "name": intent_result.name,
                "confidence": intent_result.confidence,
                "is_fallback": intent_result.is_fallback,
            },
            entities=[
                {
                    "text": e.text,
                    "type": e.type,
                    "value": e.value,
                    "confidence": e.confidence,
                    "start": e.start_pos,
                    "end": e.end_pos,
                }
                for e in entities
            ],
            sentiment={
                "label": sentiment_result[0].value,
                "score": sentiment_result[1],
            },
            processing_time_ms=processing_time,
        )
        
    except Exception as e:
        logger.error("analyze_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing text: {str(e)}",
        )


@router.get("/intents")
async def list_intents() -> dict:
    """List all available intents."""
    from nexus.data.intents import get_intent_patterns
    
    patterns = get_intent_patterns()
    
    return {
        "intents": [
            {
                "name": name,
                "description": data.get("description", ""),
                "example_patterns": data["patterns"][:3],
            }
            for name, data in patterns.items()
        ],
        "total": len(patterns),
    }


@router.get("/stats")
async def get_stats(
    engine: ConversationEngine = Depends(get_engine),
) -> dict:
    """Get engine statistics."""
    return {
        "active_sessions": len(engine._sessions),
        "engine_initialized": engine.is_initialized,
        "version": "2.0.0",
    }
