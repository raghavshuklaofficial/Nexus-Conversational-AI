"""
Conversation Engine Tests
=========================
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestConversationSession:
    """Tests for ConversationSession."""
    
    def test_session_creation(self):
        """Test session creation with defaults."""
        from nexus.core.session import ConversationSession
        
        session = ConversationSession()
        
        assert session.id is not None
        assert session.turn_count == 0
        assert session.is_expired is False
    
    def test_session_with_custom_id(self):
        """Test session creation with custom ID."""
        from uuid import uuid4
        from nexus.core.session import ConversationSession
        
        custom_id = uuid4()
        session = ConversationSession(session_id=custom_id)
        
        assert session.id == custom_id
    
    def test_session_history_management(self):
        """Test conversation history management."""
        from nexus.core.session import ConversationSession
        
        session = ConversationSession(max_history=5)
        
        assert len(session.history) == 0
        assert session.get_recent_history(3) == []
    
    def test_session_context(self):
        """Test session context tracking."""
        from nexus.core.session import ConversationSession, ConversationContext
        
        session = ConversationSession()
        
        assert isinstance(session.context, ConversationContext)
        assert session.context.last_intent is None
    
    def test_session_metrics(self):
        """Test session metrics tracking."""
        from nexus.core.session import ConversationSession, SessionMetrics
        
        session = ConversationSession()
        
        assert isinstance(session.metrics, SessionMetrics)
        assert session.metrics.total_turns == 0
    
    def test_session_export(self):
        """Test session export functionality."""
        from nexus.core.session import ConversationSession
        
        session = ConversationSession()
        export_data = session.export()
        
        assert "id" in export_data
        assert "created_at" in export_data
        assert "turn_count" in export_data


class TestEntityMemory:
    """Tests for EntityMemory."""
    
    def test_entity_storage(self):
        """Test entity storage in memory."""
        from nexus.core.session import EntityMemory
        from nexus.core.response import Entity
        
        memory = EntityMemory()
        
        entity = Entity(
            text="Paris",
            type="LOCATION",
            confidence=0.95,
            start_pos=0,
            end_pos=5,
        )
        
        memory.add(entity)
        
        assert len(memory.entities) == 1
        assert "LOCATION" in memory.entities
    
    def test_get_by_type(self):
        """Test retrieving entities by type."""
        from nexus.core.session import EntityMemory
        from nexus.core.response import Entity
        
        memory = EntityMemory()
        
        memory.add(Entity(text="Paris", type="LOCATION", confidence=0.9, start_pos=0, end_pos=5))
        memory.add(Entity(text="London", type="LOCATION", confidence=0.85, start_pos=10, end_pos=16))
        
        locations = memory.get_by_type("LOCATION")
        
        assert len(locations) == 2
    
    def test_get_latest(self):
        """Test getting most recent entity."""
        from nexus.core.session import EntityMemory
        from nexus.core.response import Entity
        
        memory = EntityMemory()
        
        memory.add(Entity(text="Paris", type="LOCATION", confidence=0.9, start_pos=0, end_pos=5))
        memory.add(Entity(text="London", type="LOCATION", confidence=0.85, start_pos=10, end_pos=16))
        
        latest = memory.get_latest("LOCATION")
        
        assert latest.text == "London"


class TestConversationContext:
    """Tests for ConversationContext."""
    
    def test_context_initialization(self):
        """Test context initialization."""
        from nexus.core.session import ConversationContext
        
        context = ConversationContext()
        
        assert context.last_intent is None
        assert len(context.active_topics) == 0
        assert context.clarification_attempts == 0
    
    def test_dominant_sentiment(self):
        """Test dominant sentiment calculation."""
        from nexus.core.session import ConversationContext
        from nexus.core.response import Sentiment
        
        context = ConversationContext()
        context.sentiment_history = [
            Sentiment.POSITIVE,
            Sentiment.POSITIVE,
            Sentiment.NEUTRAL,
        ]
        
        dominant = context.dominant_sentiment
        
        assert dominant == Sentiment.POSITIVE


class TestConversationEngine:
    """Tests for ConversationEngine."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        from nexus.config import NexusConfig
        
        with patch('nexus.config.NexusConfig') as mock:
            config = mock.return_value
            config.environment.value = "development"
            config.nlu = MagicMock()
            config.dialogue = MagicMock()
            config.app_name = "Test"
            return config
    
    def test_engine_creation(self, mock_config):
        """Test engine creation."""
        from nexus.core.engine import ConversationEngine
        
        with patch('nexus.config.get_config', return_value=mock_config):
            engine = ConversationEngine(mock_config)
            
            assert engine.is_initialized is False
            assert len(engine._sessions) == 0
    
    def test_input_preprocessing(self, mock_config):
        """Test input preprocessing."""
        from nexus.core.engine import ConversationEngine
        
        with patch('nexus.config.get_config', return_value=mock_config):
            engine = ConversationEngine(mock_config)
            
            # Test whitespace normalization
            result = engine._preprocess("  Hello   World  ")
            assert result == "Hello World"
            
            # Test stripping
            result = engine._preprocess("\n\t Test \n")
            assert result == "Test"


class TestHandlers:
    """Tests for intent handlers."""
    
    @pytest.mark.asyncio
    async def test_greeting_handler(self):
        """Test greeting handler response."""
        from nexus.dialogue.handlers import GreetingHandler
        from nexus.dialogue.state import DialogueState
        from nexus.core.response import IntentMatch
        
        handler = GreetingHandler()
        
        state = DialogueState(
            user_input="Hello",
            intent=IntentMatch(name="greeting", confidence=0.9, is_fallback=False),
        )
        
        response = await handler.handle(state)
        
        assert "text" in response
        assert len(response["text"]) > 0
    
    @pytest.mark.asyncio
    async def test_fallback_handler(self):
        """Test fallback handler response."""
        from nexus.dialogue.handlers import FallbackHandler
        from nexus.dialogue.state import DialogueState
        from nexus.core.response import IntentMatch
        
        handler = FallbackHandler()
        
        state = DialogueState(
            user_input="xyz123",
            intent=IntentMatch(name="fallback", confidence=0.1, is_fallback=True),
        )
        
        response = await handler.handle(state)
        
        assert "text" in response
        assert response["type"] in ["clarification", "fallback"]
    
    def test_handler_registry(self):
        """Test handler registration."""
        from nexus.dialogue.handlers import HandlerRegistry, GreetingHandler
        
        registry = HandlerRegistry()
        handler = GreetingHandler()
        
        registry.register(handler)
        
        assert "greeting" in registry._handlers
        assert len(registry._handlers["greeting"]) == 1
