"""
Conversation Engine
===================

The main orchestrator for the Nexus Conversational AI system.
Coordinates NLU, dialogue management, and response generation.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any
from uuid import UUID

import structlog

from nexus.config import NexusConfig, get_config
from nexus.core.response import (
    Entity,
    IntentMatch,
    Response,
    ResponseMetadata,
    ResponseType,
    Sentiment,
)
from nexus.core.session import ConversationSession
from nexus.dialogue.manager import DialogueManager
from nexus.dialogue.state import DialogueState
from nexus.nlu.classifier import IntentClassifier
from nexus.nlu.extractor import EntityExtractor
from nexus.nlu.sentiment import SentimentAnalyzer
from nexus.nlu.embeddings import EmbeddingEngine

logger = structlog.get_logger(__name__)


class ConversationEngine:
    """
    Main conversation processing engine.
    
    Orchestrates the NLU pipeline, dialogue management, and response
    generation for natural, context-aware conversations.
    
    Architecture:
        1. Input preprocessing and normalization
        2. Intent classification with confidence scoring
        3. Entity extraction and resolution
        4. Sentiment analysis
        5. Dialogue state management
        6. Response selection and generation
        7. Post-processing and formatting
    
    Example:
        >>> engine = ConversationEngine()
        >>> await engine.initialize()
        >>> response = await engine.process("Book a flight to Paris")
        >>> print(response.text)
        >>> print(response.metadata.detected_intent)
    """
    
    def __init__(self, config: NexusConfig | None = None) -> None:
        """
        Initialize the conversation engine.
        
        Args:
            config: Optional configuration. Uses default if not provided.
        """
        self.config = config or get_config()
        self._initialized = False
        
        # NLU components
        self._intent_classifier: IntentClassifier | None = None
        self._entity_extractor: EntityExtractor | None = None
        self._sentiment_analyzer: SentimentAnalyzer | None = None
        self._embedding_engine: EmbeddingEngine | None = None
        
        # Dialogue management
        self._dialogue_manager: DialogueManager | None = None
        
        # Session management
        self._sessions: dict[UUID, ConversationSession] = {}
        self._session_lock = asyncio.Lock()
        
        logger.info(
            "conversation_engine_created",
            environment=self.config.environment.value,
        )
    
    async def initialize(self) -> None:
        """
        Initialize all NLU components and load models.
        
        This method should be called before processing any inputs.
        It loads transformer models and prepares the pipeline.
        """
        if self._initialized:
            logger.warning("engine_already_initialized")
            return
        
        logger.info("initializing_conversation_engine")
        start_time = time.time()
        
        try:
            # Initialize NLU components in parallel where possible
            self._intent_classifier = IntentClassifier(self.config.nlu)
            self._entity_extractor = EntityExtractor(self.config.nlu)
            self._sentiment_analyzer = SentimentAnalyzer(self.config.nlu)
            self._embedding_engine = EmbeddingEngine(self.config.nlu)
            
            # Initialize dialogue manager
            self._dialogue_manager = DialogueManager(
                config=self.config.dialogue,
                embedding_engine=self._embedding_engine,
            )
            
            # Load models
            await asyncio.gather(
                self._intent_classifier.load(),
                self._entity_extractor.load(),
                self._sentiment_analyzer.load(),
                self._embedding_engine.load(),
            )
            
            # Load dialogue data
            await self._dialogue_manager.load()
            
            self._initialized = True
            
            elapsed = (time.time() - start_time) * 1000
            logger.info(
                "conversation_engine_initialized",
                elapsed_ms=round(elapsed, 2),
            )
            
        except Exception as e:
            logger.error("engine_initialization_failed", error=str(e))
            raise
    
    async def process(
        self,
        user_input: str,
        session: ConversationSession | None = None,
    ) -> Response:
        """
        Process user input and generate a response.
        
        Args:
            user_input: The user's message text
            session: Optional conversation session for context
        
        Returns:
            Response: Complete response with text and metadata
        
        Raises:
            RuntimeError: If engine is not initialized
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized. Call initialize() first.")
        
        start_time = time.time()
        nlu_start = start_time
        
        # Preprocess input
        processed_input = self._preprocess(user_input)
        
        if not processed_input.strip():
            return self._create_empty_input_response(start_time)
        
        try:
            # Run NLU pipeline
            intent_result, entities, sentiment_result = await asyncio.gather(
                self._intent_classifier.classify(processed_input),  # type: ignore
                self._entity_extractor.extract(processed_input),  # type: ignore
                self._sentiment_analyzer.analyze(processed_input),  # type: ignore
            )
            
            nlu_time = (time.time() - nlu_start) * 1000
            
            # Build dialogue state
            generation_start = time.time()
            
            dialogue_state = DialogueState(
                user_input=processed_input,
                intent=intent_result,
                entities=entities,
                sentiment=sentiment_result,
                session=session,
            )
            
            # Generate response through dialogue manager
            response_data = await self._dialogue_manager.generate_response(  # type: ignore
                dialogue_state
            )
            
            generation_time = (time.time() - generation_start) * 1000
            total_time = (time.time() - start_time) * 1000
            
            # Build response object
            response = self._build_response(
                response_data=response_data,
                intent_result=intent_result,
                entities=entities,
                sentiment_result=sentiment_result,
                session=session,
                processing_time_ms=total_time,
                nlu_time_ms=nlu_time,
                generation_time_ms=generation_time,
            )
            
            logger.info(
                "message_processed",
                intent=intent_result.name,
                confidence=round(intent_result.confidence, 3),
                processing_ms=round(total_time, 2),
            )
            
            return response
            
        except Exception as e:
            logger.error("processing_error", error=str(e), input=user_input[:100])
            return self._create_error_response(str(e), start_time)
    
    def _preprocess(self, text: str) -> str:
        """Preprocess and normalize user input."""
        # Remove excessive whitespace
        text = " ".join(text.split())
        # Basic normalization
        text = text.strip()
        return text
    
    def _build_response(
        self,
        response_data: dict[str, Any],
        intent_result: IntentMatch,
        entities: list[Entity],
        sentiment_result: tuple[Sentiment, float],
        session: ConversationSession | None,
        processing_time_ms: float,
        nlu_time_ms: float,
        generation_time_ms: float,
    ) -> Response:
        """Build a complete Response object."""
        sentiment, sentiment_score = sentiment_result
        
        metadata = ResponseMetadata(
            processing_time_ms=processing_time_ms,
            nlu_time_ms=nlu_time_ms,
            generation_time_ms=generation_time_ms,
            detected_intent=intent_result,
            extracted_entities=entities,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            dialogue_turn=session.turn_count + 1 if session else 1,
            session_id=session.id if session else None,
            model_version=self.config.app_name,
            nlu_model=self.config.nlu.intent_model,
        )
        
        return Response(
            text=response_data.get("text", ""),
            type=ResponseType(response_data.get("type", "standard")),
            suggestions=response_data.get("suggestions", []),
            actions=response_data.get("actions", []),
            metadata=metadata,
        )
    
    def _create_empty_input_response(self, start_time: float) -> Response:
        """Create response for empty input."""
        return Response(
            text="I didn't catch that. Could you please say something?",
            type=ResponseType.CLARIFICATION,
            metadata=ResponseMetadata(
                processing_time_ms=(time.time() - start_time) * 1000,
            ),
        )
    
    def _create_error_response(self, error: str, start_time: float) -> Response:
        """Create response for processing errors."""
        return Response(
            text="I apologize, but I encountered an issue processing your request. Please try again.",
            type=ResponseType.ERROR,
            metadata=ResponseMetadata(
                processing_time_ms=(time.time() - start_time) * 1000,
            ),
        )
    
    async def create_session(self) -> ConversationSession:
        """
        Create a new conversation session.
        
        Returns:
            ConversationSession: New session bound to this engine
        """
        async with self._session_lock:
            session = ConversationSession(
                max_history=self.config.dialogue.max_history_turns,
                timeout_minutes=self.config.dialogue.session_timeout_minutes,
            )
            session.bind_engine(self)
            self._sessions[session.id] = session
            
            logger.info("session_created", session_id=str(session.id))
            return session
    
    async def get_session(self, session_id: UUID) -> ConversationSession | None:
        """Get an existing session by ID."""
        return self._sessions.get(session_id)
    
    async def cleanup_sessions(self) -> int:
        """
        Remove expired sessions.
        
        Returns:
            int: Number of sessions removed
        """
        async with self._session_lock:
            expired = [
                sid for sid, session in self._sessions.items()
                if session.is_expired
            ]
            for sid in expired:
                del self._sessions[sid]
            
            if expired:
                logger.info("sessions_cleaned", count=len(expired))
            
            return len(expired)
    
    @classmethod
    async def from_pretrained(
        cls,
        model_path: str | Path,
        config: NexusConfig | None = None,
    ) -> ConversationEngine:
        """
        Load a pre-trained conversation engine.
        
        Args:
            model_path: Path to the model directory
            config: Optional configuration override
        
        Returns:
            ConversationEngine: Initialized engine
        """
        engine = cls(config)
        # Load custom model weights if available
        await engine.initialize()
        return engine
    
    @property
    def is_initialized(self) -> bool:
        """Check if the engine is initialized."""
        return self._initialized
    
    def __repr__(self) -> str:
        return (
            f"ConversationEngine(initialized={self._initialized}, "
            f"sessions={len(self._sessions)})"
        )
