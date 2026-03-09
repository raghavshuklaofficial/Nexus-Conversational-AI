"""
Session management for individual conversations — tracks history,
context, entity memory, and metrics per session.
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, PrivateAttr

from nexus.core.response import ConversationTurn, Entity, Response, Sentiment

if TYPE_CHECKING:
    from nexus.core.engine import ConversationEngine


class SessionMetrics(BaseModel):
    """Tracks stats for a conversation session."""
    
    total_turns: int = Field(default=0, ge=0)
    total_processing_time_ms: float = Field(default=0.0, ge=0)
    average_response_time_ms: float = Field(default=0.0, ge=0)
    fallback_count: int = Field(default=0, ge=0)
    clarification_count: int = Field(default=0, ge=0)
    sentiment_trend: list[float] = Field(default_factory=list)
    
    def update(self, response: Response) -> None:
        """Update metrics with a new response."""
        self.total_turns += 1
        self.total_processing_time_ms += response.metadata.processing_time_ms
        self.average_response_time_ms = self.total_processing_time_ms / self.total_turns
        self.sentiment_trend.append(response.metadata.sentiment_score)


class EntityMemory(BaseModel):
    """Stores extracted entities within a session so we can reference them later."""
    
    entities: dict[str, list[Entity]] = Field(default_factory=dict)
    
    def add(self, entity: Entity) -> None:
        """Add entity to memory, skipping duplicates."""
        if entity.type not in self.entities:
            self.entities[entity.type] = []
        
        # Avoid duplicates
        existing_values = {e.value or e.text for e in self.entities[entity.type]}
        if (entity.value or entity.text) not in existing_values:
            self.entities[entity.type].append(entity)
    
    def get_by_type(self, entity_type: str) -> list[Entity]:
        """Get all entities of a specific type."""
        return self.entities.get(entity_type, [])
    
    def get_latest(self, entity_type: str) -> Entity | None:
        """Get the most recently added entity of a type."""
        entities = self.entities.get(entity_type, [])
        return entities[-1] if entities else None
    
    def clear(self) -> None:
        """Clear all stored entities."""
        self.entities.clear()


class ConversationContext(BaseModel):
    """Tracks topics, entities, preferences and sentiment throughout a conversation."""
    
    # Active topics being discussed
    active_topics: list[str] = Field(default_factory=list)
    
    # Entity memory
    entity_memory: EntityMemory = Field(default_factory=EntityMemory)
    
    # Last detected intent
    last_intent: str | None = Field(default=None)
    
    # Clarification tracking
    pending_clarification: str | None = Field(default=None)
    clarification_attempts: int = Field(default=0, ge=0)
    
    # User preferences detected during conversation
    preferences: dict[str, Any] = Field(default_factory=dict)
    
    # Overall sentiment trend
    sentiment_history: list[Sentiment] = Field(default_factory=list)
    
    @property
    def dominant_sentiment(self) -> Sentiment:
        """Calculate the dominant sentiment from history."""
        if not self.sentiment_history:
            return Sentiment.NEUTRAL
        
        sentiment_scores = {
            Sentiment.VERY_NEGATIVE: -2,
            Sentiment.NEGATIVE: -1,
            Sentiment.NEUTRAL: 0,
            Sentiment.POSITIVE: 1,
            Sentiment.VERY_POSITIVE: 2,
        }
        
        avg_score = sum(sentiment_scores[s] for s in self.sentiment_history) / len(self.sentiment_history)
        
        if avg_score <= -1.5:
            return Sentiment.VERY_NEGATIVE
        elif avg_score <= -0.5:
            return Sentiment.NEGATIVE
        elif avg_score < 0.5:
            return Sentiment.NEUTRAL
        elif avg_score < 1.5:
            return Sentiment.POSITIVE
        return Sentiment.VERY_POSITIVE
    
    def update_from_response(self, response: Response) -> None:
        """Update context based on a response."""
        # Update sentiment history
        self.sentiment_history.append(response.metadata.sentiment)
        if len(self.sentiment_history) > 20:  # don't let this grow forever
            self.sentiment_history.pop(0)
        
        # Update intent
        if response.metadata.detected_intent:
            self.last_intent = response.metadata.detected_intent.name
        
        # Store entities
        for entity in response.metadata.extracted_entities:
            self.entity_memory.add(entity)


class ConversationSession:
    """Manages a single conversation — history, context, metrics."""
    
    def __init__(
        self,
        session_id: UUID | None = None,
        max_history: int = 50,
        timeout_minutes: int = 30,
    ) -> None:
        self.id = session_id or uuid4()
        self.created_at = datetime.utcnow()
        self.last_activity = self.created_at
        self.timeout = timedelta(minutes=timeout_minutes)
        
        self._max_history = max_history
        self._history: deque[ConversationTurn] = deque(maxlen=max_history)
        self._context = ConversationContext()
        self._metrics = SessionMetrics()
        self._engine: ConversationEngine | None = None
        self._lock = asyncio.Lock()
    
    @property
    def history(self) -> list[ConversationTurn]:
        """Get conversation history as a list."""
        return list(self._history)
    
    @property
    def context(self) -> ConversationContext:
        """Get the conversation context."""
        return self._context
    
    @property
    def metrics(self) -> SessionMetrics:
        """Get session metrics."""
        return self._metrics
    
    @property
    def is_expired(self) -> bool:
        """Check if the session has expired."""
        return datetime.utcnow() - self.last_activity > self.timeout
    
    @property
    def turn_count(self) -> int:
        """Get the number of conversation turns."""
        return len(self._history)
    
    def bind_engine(self, engine: ConversationEngine) -> None:
        """Bind a conversation engine to this session."""
        self._engine = engine
    
    async def process(self, user_input: str) -> Response:
        """Process a message in this session's context."""
        if self._engine is None:
            raise RuntimeError("No conversation engine bound to session")
        
        async with self._lock:
            # Check expiration
            if self.is_expired:
                self._reset_context()
            
            # Process through engine
            response = await self._engine.process(
                user_input,
                session=self,
            )
            
            # Record the turn
            turn = ConversationTurn(
                user_input=user_input,
                response=response,
            )
            self._history.append(turn)
            
            # Update metrics and context
            self._metrics.update(response)
            self._context.update_from_response(response)
            self.last_activity = datetime.utcnow()
            
            return response
    
    def get_recent_history(self, n_turns: int = 5) -> list[ConversationTurn]:
        """Get the n most recent conversation turns."""
        return list(self._history)[-n_turns:]
    
    def get_history_text(self, n_turns: int = 5) -> str:
        """Get recent history as formatted text."""
        turns = self.get_recent_history(n_turns)
        lines = []
        for turn in turns:
            lines.append(f"User: {turn.user_input}")
            lines.append(f"Assistant: {turn.response.text}")
        return "\n".join(lines)
    
    def _reset_context(self) -> None:
        """Reset session context (on expiration)."""
        self._context = ConversationContext()
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history.clear()
    
    def export(self) -> dict[str, Any]:
        """Export session data for persistence."""
        return {
            "id": str(self.id),
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "turn_count": self.turn_count,
            "history": [
                {
                    "user_input": turn.user_input,
                    "response_text": turn.response.text,
                    "timestamp": turn.timestamp.isoformat(),
                }
                for turn in self._history
            ],
            "metrics": self._metrics.model_dump(),
        }
    
    def __repr__(self) -> str:
        return (
            f"ConversationSession(id={self.id}, "
            f"turns={self.turn_count}, "
            f"created={self.created_at.isoformat()})"
        )
