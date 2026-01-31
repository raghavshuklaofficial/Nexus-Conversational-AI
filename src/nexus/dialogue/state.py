"""
Dialogue State
==============

State representation for dialogue management with context tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from nexus.core.response import Entity, IntentMatch, Sentiment

if TYPE_CHECKING:
    from nexus.core.session import ConversationSession


@dataclass
class DialogueState:
    """
    Complete state representation for a dialogue turn.
    
    Encapsulates all information needed for response generation,
    including NLU results, context, and session data.
    
    Attributes:
        user_input: The original user message
        intent: Classified intent with confidence
        entities: Extracted entities
        sentiment: Analyzed sentiment and score
        session: Active conversation session
        context: Additional context information
        timestamp: State creation time
    """
    
    user_input: str
    intent: IntentMatch
    entities: list[Entity] = field(default_factory=list)
    sentiment: tuple[Sentiment, float] = field(
        default=(Sentiment.NEUTRAL, 0.0)
    )
    session: ConversationSession | None = None
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def has_entities(self) -> bool:
        """Check if any entities were extracted."""
        return len(self.entities) > 0
    
    @property
    def is_positive_sentiment(self) -> bool:
        """Check if sentiment is positive."""
        return self.sentiment[1] > 0.2
    
    @property
    def is_negative_sentiment(self) -> bool:
        """Check if sentiment is negative."""
        return self.sentiment[1] < -0.2
    
    def get_entity(self, entity_type: str) -> Entity | None:
        """Get the first entity of a specific type."""
        for entity in self.entities:
            if entity.type == entity_type:
                return entity
        return None
    
    def get_entities(self, entity_type: str) -> list[Entity]:
        """Get all entities of a specific type."""
        return [e for e in self.entities if e.type == entity_type]
    
    @property
    def turn_number(self) -> int:
        """Get the current turn number in the conversation."""
        if self.session:
            return self.session.turn_count + 1
        return 1
    
    @property
    def conversation_history(self) -> str:
        """Get recent conversation history as text."""
        if self.session:
            return self.session.get_history_text(n_turns=3)
        return ""
    
    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "user_input": self.user_input,
            "intent": {
                "name": self.intent.name,
                "confidence": self.intent.confidence,
            },
            "entities": [
                {
                    "text": e.text,
                    "type": e.type,
                    "value": e.value,
                }
                for e in self.entities
            ],
            "sentiment": {
                "label": self.sentiment[0].value,
                "score": self.sentiment[1],
            },
            "turn_number": self.turn_number,
            "timestamp": self.timestamp.isoformat(),
        }
