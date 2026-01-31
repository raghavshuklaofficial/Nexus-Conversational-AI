"""
Nexus Conversational AI Engine
==============================

Enterprise-grade conversational AI system with transformer-based NLU,
multi-turn dialogue management, and real-time analytics.

Features:
    - Transformer-based intent classification and entity extraction
    - Semantic similarity matching with sentence embeddings
    - Multi-turn dialogue state tracking
    - Context-aware response generation
    - Real-time sentiment analysis
    - Prometheus metrics and observability
    - WebSocket and REST API support

Example:
    >>> from nexus import ConversationEngine
    >>> engine = ConversationEngine.from_pretrained("nexus-base")
    >>> response = await engine.process("Hello, how are you?")
    >>> print(response.text)

"""

from nexus.core.engine import ConversationEngine
from nexus.core.session import ConversationSession
from nexus.nlu.classifier import IntentClassifier
from nexus.nlu.extractor import EntityExtractor
from nexus.dialogue.manager import DialogueManager
from nexus.dialogue.state import DialogueState

__version__ = "2.0.0"
__author__ = "Raghav Shukla"

__all__ = [
    "ConversationEngine",
    "ConversationSession",
    "IntentClassifier",
    "EntityExtractor",
    "DialogueManager",
    "DialogueState",
    "__version__",
]
