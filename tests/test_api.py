"""
API Integration Tests
=====================
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestHealthEndpoint:
    """Tests for the health check endpoint."""
    
    @pytest.mark.asyncio
    async def test_health_check_returns_healthy(self):
        """Test that health check returns healthy status."""
        from fastapi.testclient import TestClient
        from nexus.api.app import create_app
        
        app = create_app()
        
        # Mock the lifespan
        with patch('nexus.api.app.ConversationEngine') as mock_engine:
            mock_engine.return_value.is_initialized = True
            
            client = TestClient(app)
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"


class TestChatEndpoint:
    """Tests for the chat endpoint."""
    
    @pytest.fixture
    def mock_engine(self):
        """Create a mock conversation engine."""
        engine = MagicMock()
        engine.is_initialized = True
        return engine
    
    def test_chat_request_validation(self):
        """Test that chat endpoint validates requests."""
        from nexus.api.routes import MessageRequest
        
        # Valid request
        request = MessageRequest(text="Hello")
        assert request.text == "Hello"
        
        # Test with session_id
        from uuid import uuid4
        session_id = uuid4()
        request = MessageRequest(text="Hello", session_id=session_id)
        assert request.session_id == session_id
    
    def test_message_response_model(self):
        """Test MessageResponse model structure."""
        from nexus.api.routes import MessageResponse
        from uuid import uuid4
        
        response = MessageResponse(
            id=str(uuid4()),
            text="Hello!",
            type="standard",
            session_id=str(uuid4()),
            suggestions=["Ask more"],
            sentiment="positive",
            confidence=0.9,
            intent="greeting",
            entities=[],
            processing_time_ms=50.0,
            timestamp="2025-01-01T00:00:00",
        )
        
        assert response.text == "Hello!"
        assert response.confidence == 0.9


class TestSessionEndpoints:
    """Tests for session management endpoints."""
    
    def test_session_response_model(self):
        """Test SessionResponse model structure."""
        from nexus.api.routes import SessionResponse
        from uuid import uuid4
        
        response = SessionResponse(
            session_id=str(uuid4()),
            created_at="2025-01-01T00:00:00",
            turn_count=5,
            is_expired=False,
        )
        
        assert response.turn_count == 5
        assert response.is_expired is False


class TestAnalyzeEndpoint:
    """Tests for the analyze endpoint."""
    
    def test_analyze_request_validation(self):
        """Test that analyze endpoint validates requests."""
        from nexus.api.routes import AnalyzeRequest
        
        request = AnalyzeRequest(text="Test message")
        assert request.text == "Test message"
    
    def test_analyze_response_model(self):
        """Test AnalyzeResponse model structure."""
        from nexus.api.routes import AnalyzeResponse
        
        response = AnalyzeResponse(
            intent={"name": "greeting", "confidence": 0.9},
            entities=[{"text": "John", "type": "PERSON"}],
            sentiment={"label": "positive", "score": 0.8},
            processing_time_ms=25.0,
        )
        
        assert response.intent["name"] == "greeting"
        assert len(response.entities) == 1


class TestWebSocket:
    """Tests for WebSocket functionality."""
    
    def test_websocket_message_model(self):
        """Test WebSocketMessage model."""
        from nexus.api.websocket import WebSocketMessage
        
        message = WebSocketMessage(
            type="message",
            payload={"text": "Hello"},
        )
        
        assert message.type == "message"
        assert message.payload["text"] == "Hello"
        assert message.timestamp  # Should be auto-set
    
    def test_connection_manager_initialization(self):
        """Test ConnectionManager initialization."""
        from nexus.api.websocket import ConnectionManager
        
        manager = ConnectionManager()
        
        assert manager.connection_count == 0


class TestRateLimiting:
    """Tests for rate limiting functionality."""
    
    def test_rate_limit_config(self):
        """Test rate limit configuration."""
        from nexus.config import APIConfig
        
        config = APIConfig()
        
        assert config.rate_limit_per_minute > 0
        assert config.rate_limit_per_minute == 60  # default


class TestCORSConfiguration:
    """Tests for CORS configuration."""
    
    def test_cors_enabled_by_default(self):
        """Test that CORS is enabled by default."""
        from nexus.config import APIConfig
        
        config = APIConfig()
        
        assert config.enable_cors is True
        assert "*" in config.allowed_origins
