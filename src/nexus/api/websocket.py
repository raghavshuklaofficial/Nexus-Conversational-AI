"""
WebSocket handler for real-time chat. Manages connections,
heartbeats, and message routing.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

import structlog
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, ValidationError

from nexus.core.engine import ConversationEngine
from nexus.core.session import ConversationSession

logger = structlog.get_logger(__name__)

websocket_router = APIRouter()


class WebSocketMessage(BaseModel):
    """Model for WebSocket messages."""
    
    type: str
    payload: dict[str, Any] = {}
    timestamp: str = ""
    
    def __init__(self, **data: Any) -> None:
        if "timestamp" not in data or not data["timestamp"]:
            data["timestamp"] = datetime.utcnow().isoformat()
        super().__init__(**data)


class ConnectionManager:
    """Tracks active WebSocket connections and their sessions."""
    
    def __init__(self) -> None:
        self._connections: dict[UUID, WebSocket] = {}
        self._sessions: dict[UUID, ConversationSession] = {}
        self._heartbeat_tasks: dict[UUID, asyncio.Task] = {}
    
    async def connect(
        self,
        websocket: WebSocket,
        engine: ConversationEngine,
    ) -> tuple[UUID, ConversationSession]:
        """
        Accept a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection
            engine: Conversation engine
        
        Returns:
            tuple: Connection ID and session
        """
        await websocket.accept()
        
        connection_id = uuid4()
        session = await engine.create_session()
        
        self._connections[connection_id] = websocket
        self._sessions[connection_id] = session
        
        # Start heartbeat
        self._heartbeat_tasks[connection_id] = asyncio.create_task(
            self._heartbeat(connection_id)
        )
        
        logger.info(
            "websocket_connected",
            connection_id=str(connection_id),
            session_id=str(session.id),
        )
        
        # Send connection confirmation
        await self.send_message(connection_id, WebSocketMessage(
            type="connected",
            payload={
                "connection_id": str(connection_id),
                "session_id": str(session.id),
            },
        ))
        
        return connection_id, session
    
    async def disconnect(self, connection_id: UUID) -> None:
        """
        Handle WebSocket disconnection.
        
        Args:
            connection_id: Connection to disconnect
        """
        # Cancel heartbeat
        if connection_id in self._heartbeat_tasks:
            self._heartbeat_tasks[connection_id].cancel()
            del self._heartbeat_tasks[connection_id]
        
        # Remove connection
        if connection_id in self._connections:
            del self._connections[connection_id]
        
        if connection_id in self._sessions:
            del self._sessions[connection_id]
        
        logger.info("websocket_disconnected", connection_id=str(connection_id))
    
    async def send_message(
        self,
        connection_id: UUID,
        message: WebSocketMessage,
    ) -> None:
        """
        Send a message to a specific connection.
        
        Args:
            connection_id: Target connection
            message: Message to send
        """
        if connection_id in self._connections:
            websocket = self._connections[connection_id]
            await websocket.send_json(message.model_dump())
    
    async def broadcast(self, message: WebSocketMessage) -> None:
        """
        Broadcast a message to all connections.
        
        Args:
            message: Message to broadcast
        """
        for connection_id in list(self._connections.keys()):
            try:
                await self.send_message(connection_id, message)
            except Exception as e:
                logger.error(
                    "broadcast_error",
                    connection_id=str(connection_id),
                    error=str(e),
                )
    
    async def _heartbeat(self, connection_id: UUID) -> None:
        """Send periodic heartbeat messages."""
        try:
            while True:
                await asyncio.sleep(30)
                
                if connection_id in self._connections:
                    await self.send_message(connection_id, WebSocketMessage(
                        type="heartbeat",
                        payload={"status": "alive"},
                    ))
        except asyncio.CancelledError:
            pass
    
    def get_session(self, connection_id: UUID) -> ConversationSession | None:
        """Get the session for a connection."""
        return self._sessions.get(connection_id)
    
    @property
    def connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self._connections)


# Global connection manager
manager = ConnectionManager()


@websocket_router.websocket("/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time chat.
    
    Message Protocol:
    
    Client -> Server:
        - {"type": "message", "payload": {"text": "Hello"}}
        - {"type": "ping"}
    
    Server -> Client:
        - {"type": "connected", "payload": {"session_id": "..."}}
        - {"type": "response", "payload": {...}}
        - {"type": "typing", "payload": {"is_typing": true}}
        - {"type": "heartbeat"}
        - {"type": "error", "payload": {"message": "..."}}
    """
    # Get engine from app state
    engine: ConversationEngine = websocket.app.state.engine
    
    connection_id: UUID | None = None
    
    try:
        connection_id, session = await manager.connect(websocket, engine)
        
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            try:
                message = WebSocketMessage(**data)
            except ValidationError as e:
                await manager.send_message(connection_id, WebSocketMessage(
                    type="error",
                    payload={"message": "Invalid message format", "details": str(e)},
                ))
                continue
            
            # Handle message types
            if message.type == "message":
                await _handle_chat_message(connection_id, message, session, engine, websocket)
            
            elif message.type == "ping":
                await manager.send_message(connection_id, WebSocketMessage(
                    type="pong",
                    payload={},
                ))
            
            elif message.type == "clear_history":
                session.clear_history()
                await manager.send_message(connection_id, WebSocketMessage(
                    type="history_cleared",
                    payload={},
                ))
            
            else:
                await manager.send_message(connection_id, WebSocketMessage(
                    type="error",
                    payload={"message": f"Unknown message type: {message.type}"},
                ))
    
    except WebSocketDisconnect:
        logger.info("client_disconnected", connection_id=str(connection_id))
    
    except Exception as e:
        logger.error("websocket_error", error=str(e))
    
    finally:
        if connection_id:
            await manager.disconnect(connection_id)


async def _handle_chat_message(
    connection_id: UUID,
    message: WebSocketMessage,
    session: ConversationSession,
    engine: ConversationEngine,
    websocket: WebSocket | None = None,
) -> None:
    """Handle incoming chat message via the new ChatService if available."""
    text = message.payload.get("text", "").strip()
    use_rag = message.payload.get("use_rag", False)
    
    if not text:
        await manager.send_message(connection_id, WebSocketMessage(
            type="error",
            payload={"message": "Empty message"},
        ))
        return
    
    # Send typing indicator
    await manager.send_message(connection_id, WebSocketMessage(
        type="typing",
        payload={"is_typing": True},
    ))
    
    try:
        # Try the new ChatService first (has LLM, RAG, etc.)
        chat_service = getattr(websocket.app.state, "chat_service", None) if websocket else None
        
        if chat_service is not None:
            from nexus.domain.models import ChatResponse
            resp: ChatResponse = await chat_service.chat(
                text=text,
                session_id=str(session.id),
                use_rag=use_rag,
                top_k=3,
            )
            
            await manager.send_message(connection_id, WebSocketMessage(
                type="typing",
                payload={"is_typing": False},
            ))
            
            await manager.send_message(connection_id, WebSocketMessage(
                type="response",
                payload={
                    "text": resp.answer,
                    "answer": resp.answer,
                    "intent": resp.intent,
                    "sentiment": resp.sentiment,
                    "sentiment_score": resp.sentiment_score,
                    "processing_time_ms": resp.latency.total_ms,
                    "latency": resp.latency.model_dump(),
                    "entities": resp.entities,
                    "citations": [c.model_dump() for c in resp.citations],
                    "cache_hit": resp.cache_hit,
                    "model_name": resp.model_name,
                    "suggestions": resp.suggestions,
                    "session_id": resp.session_id,
                },
            ))
        else:
            # Fallback to old engine
            response = await engine.process(text, session=session)
            
            await manager.send_message(connection_id, WebSocketMessage(
                type="typing",
                payload={"is_typing": False},
            ))
            
            await manager.send_message(connection_id, WebSocketMessage(
                type="response",
                payload={
                    "id": str(response.id),
                    "text": response.text,
                    "type": response.type.value,
                    "suggestions": response.suggestions,
                    "sentiment": response.metadata.sentiment.value,
                    "confidence": response.metadata.detected_intent.confidence if response.metadata.detected_intent else 0.0,
                    "intent": response.metadata.detected_intent.name if response.metadata.detected_intent else "unknown",
                    "processing_time_ms": response.metadata.processing_time_ms,
                },
            ))
    
    except Exception as e:
        logger.error("message_processing_error", error=str(e))
        
        await manager.send_message(connection_id, WebSocketMessage(
            type="typing",
            payload={"is_typing": False},
        ))
        
        await manager.send_message(connection_id, WebSocketMessage(
            type="error",
            payload={"message": "Error processing message"},
        ))
