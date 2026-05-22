# Shared fixtures for all tests

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    import asyncio
    
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_nlu_config():
    """Create mock NLU configuration."""
    config = MagicMock()
    config.intent_model = "sentence-transformers/all-MiniLM-L6-v2"
    config.entity_model = "dslim/bert-base-NER"
    config.embedding_model = "sentence-transformers/all-mpnet-base-v2"
    config.device = "cpu"
    config.fallback_threshold = 0.4
    config.max_input_length = 512
    return config


@pytest.fixture
def mock_dialogue_config():
    """Create mock dialogue configuration."""
    config = MagicMock()
    config.max_history_turns = 10
    config.context_window_size = 5
    config.session_timeout_minutes = 30
    return config


@pytest.fixture
def mock_api_config():
    """Create mock API configuration."""
    config = MagicMock()
    config.host = "0.0.0.0"
    config.port = 8000
    config.debug = True
    config.rate_limit_per_minute = 60
    config.enable_cors = True
    config.allowed_origins = ["*"]
    return config


@pytest.fixture
def mock_nexus_config(mock_nlu_config, mock_dialogue_config, mock_api_config):
    """Create complete mock Nexus configuration."""
    from enum import Enum
    
    class MockEnvironment(str, Enum):
        DEVELOPMENT = "development"
    
    config = MagicMock()
    config.environment = MockEnvironment.DEVELOPMENT
    config.app_name = "Nexus Test"
    config.nlu = mock_nlu_config
    config.dialogue = mock_dialogue_config
    config.api = mock_api_config
    config.cache = MagicMock()
    config.cache.backend = "memory"
    config.cache.default_ttl = 3600
    config.vector_store_backend = "faiss"
    config.qdrant_url = "http://localhost:6333"
    config.kafka_enabled = False
    config.llm_provider = "local"
    config.llm_model = "gpt2"
    return config


@pytest.fixture
def sample_intents():
    """Sample intent data for testing."""
    return {
        "greeting": {
            "patterns": [
                "hello", "hi there", "hey", "good morning", "good evening",
            ],
            "responses": [
                "Hello! How can I assist you today?",
                "Hi there! What can I help you with?",
            ],
        },
        "goodbye": {
            "patterns": [
                "bye", "goodbye", "see you later", "take care",
            ],
            "responses": [
                "Goodbye! Have a great day!",
                "See you later! Take care!",
            ],
        },
        "thanks": {
            "patterns": [
                "thank you", "thanks", "appreciate it",
            ],
            "responses": [
                "You're welcome!",
                "Happy to help!",
            ],
        },
    }


@pytest.fixture
def sample_entities():
    """Sample entity data for testing."""
    from nexus.core.response import Entity
    
    return [
        Entity(text="John Smith", type="PERSON", confidence=0.95, start_pos=0, end_pos=10),
        Entity(text="john@example.com", type="EMAIL", confidence=0.99, start_pos=20, end_pos=36),
        Entity(text="New York", type="LOCATION", confidence=0.92, start_pos=40, end_pos=48),
        Entity(text="tomorrow", type="DATE", confidence=0.88, start_pos=60, end_pos=68),
    ]


@pytest.fixture
def sample_messages():
    """Sample conversation messages for testing."""
    return [
        {"role": "user", "content": "Hello, I need some help"},
        {"role": "assistant", "content": "Hello! I'd be happy to help. What do you need?"},
        {"role": "user", "content": "What's the weather like today?"},
        {"role": "assistant", "content": "I can help with general questions. What would you like to know?"},
    ]


@pytest.fixture
def conversation_session(mock_dialogue_config):
    """Create a test conversation session."""
    from nexus.core.session import ConversationSession
    
    return ConversationSession(max_history=mock_dialogue_config.max_history_turns)


# Markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests requiring infrastructure")
    config.addinivalue_line("markers", "slow: Slow tests requiring model loading")
    config.addinivalue_line("markers", "e2e: End-to-end tests")


# Test environment setup
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ["NEXUS_ENVIRONMENT"] = "testing"
    os.environ["NEXUS_DEBUG"] = "true"
    os.environ["NEXUS_LOG_LEVEL"] = "DEBUG"
    os.environ["NEXUS_NLU_DEVICE"] = "cpu"
    
    yield
    
    for key in ["NEXUS_ENVIRONMENT", "NEXUS_DEBUG", "NEXUS_LOG_LEVEL", "NEXUS_NLU_DEVICE"]:
        os.environ.pop(key, None)
