"""Chat routes — POST /chat, POST /chat/stream"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from nexus.api.dependencies import get_chat_service
from nexus.api.schemas.chat import ChatRequest, ChatResponseSchema
from nexus.application.chat_service import ChatService
from nexus.infrastructure.observability.tracing import generate_request_id

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponseSchema)
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponseSchema:
    """Send a message, get a response. Optionally enable RAG."""
    req_id = generate_request_id()
    try:
        result = await chat_service.chat(
            text=request.text,
            session_id=request.session_id,
            use_rag=request.use_rag,
            top_k=request.top_k,
            request_id=req_id,
        )
        return ChatResponseSchema(**result.model_dump())
    except Exception as e:
        logger.error("chat_error", error=str(e), request_id=req_id)
        raise HTTPException(status_code=500, detail="Error processing message")


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
):
    """Streaming chat response via Server-Sent Events."""
    req_id = generate_request_id()

    async def event_stream():
        try:
            result = await chat_service.chat(
                text=request.text,
                session_id=request.session_id,
                use_rag=request.use_rag,
                top_k=request.top_k,
                request_id=req_id,
            )
            # Stream the answer word by word
            words = result.answer.split()
            for i in range(0, len(words), 3):
                chunk = " ".join(words[i:i + 3])
                yield f"data: {json.dumps({'text': chunk, 'done': False})}\n\n"
                await asyncio.sleep(0.05)

            # Final event with full metadata
            yield f"data: {json.dumps({'text': '', 'done': True, 'session_id': result.session_id, 'intent': result.intent, 'sentiment': result.sentiment})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
