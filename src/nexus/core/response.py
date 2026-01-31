"""
Response Models
===============

Structured response types for the conversation engine with rich metadata
and serialization support.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ResponseType(str, Enum):
    """Types of responses the engine can generate."""
    
    STANDARD = "standard"
    CLARIFICATION = "clarification"
    FALLBACK = "fallback"
    ERROR = "error"
    HANDOFF = "handoff"
    MULTI_PART = "multi_part"


class Sentiment(str, Enum):
    """Detected sentiment in user input."""
    
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class Entity(BaseModel):
    """Extracted entity from user input."""
    
    text: str = Field(..., description="The entity text as it appears in input")
    type: str = Field(..., description="Entity type (e.g., 'PERSON', 'DATE', 'LOCATION')")
    value: str | None = Field(default=None, description="Normalized/resolved value")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    start_pos: int = Field(..., ge=0, description="Start position in original text")
    end_pos: int = Field(..., ge=0, description="End position in original text")
    
    class Config:
        frozen = True


class IntentMatch(BaseModel):
    """Matched intent with confidence score."""
    
    name: str = Field(..., description="Intent identifier")
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_fallback: bool = Field(default=False)
    
    class Config:
        frozen = True


class ResponseMetadata(BaseModel):
    """
    Rich metadata about the response generation process.
    
    Provides transparency into the AI's decision-making for debugging,
    analytics, and improving the conversation model.
    """
    
    # Timing information
    processing_time_ms: float = Field(..., ge=0)
    nlu_time_ms: float = Field(default=0, ge=0)
    generation_time_ms: float = Field(default=0, ge=0)
    
    # NLU results
    detected_intent: IntentMatch | None = Field(default=None)
    alternative_intents: list[IntentMatch] = Field(default_factory=list)
    extracted_entities: list[Entity] = Field(default_factory=list)
    
    # Sentiment analysis
    sentiment: Sentiment = Field(default=Sentiment.NEUTRAL)
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    
    # Context information
    context_used: list[str] = Field(default_factory=list)
    dialogue_turn: int = Field(default=1, ge=1)
    session_id: UUID | None = Field(default=None)
    
    # Model information
    model_version: str = Field(default="2.0.0")
    nlu_model: str = Field(default="")
    
    class Config:
        frozen = True


class Response(BaseModel):
    """
    Complete response from the conversation engine.
    
    Encapsulates the response text, type, and all associated metadata
    for comprehensive conversation tracking.
    """
    
    id: UUID = Field(default_factory=uuid4, description="Unique response identifier")
    text: str = Field(..., description="The response text")
    type: ResponseType = Field(default=ResponseType.STANDARD)
    
    # Rich response features
    alternatives: list[str] = Field(
        default_factory=list,
        description="Alternative response options"
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Follow-up suggestions for the user"
    )
    actions: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Actionable items (buttons, links, etc.)"
    )
    
    # Metadata
    metadata: ResponseMetadata = Field(...)
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def to_simple(self) -> dict[str, Any]:
        """Return simplified response for API output."""
        return {
            "id": str(self.id),
            "text": self.text,
            "type": self.type.value,
            "suggestions": self.suggestions,
            "sentiment": self.metadata.sentiment.value,
            "confidence": self.metadata.detected_intent.confidence if self.metadata.detected_intent else 0.0,
        }
    
    class Config:
        frozen = True


class ConversationTurn(BaseModel):
    """A single turn in the conversation."""
    
    id: UUID = Field(default_factory=uuid4)
    user_input: str = Field(...)
    response: Response = Field(...)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        frozen = True
