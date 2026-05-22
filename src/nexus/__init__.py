"""
Nexus Conversational AI

Production-grade conversational AI platform with RAG,
async document ingestion, transformer NLU, and multi-turn
dialogue management.
"""

from nexus.core.engine import ConversationEngine
from nexus.core.session import ConversationSession
from nexus.nlu.classifier import IntentClassifier
from nexus.nlu.extractor import EntityExtractor
from nexus.dialogue.manager import DialogueManager
from nexus.dialogue.state import DialogueState

__version__ = "3.0.0"
__author__ = "Raghav Shukla"

__all__ = [
    "ConversationEngine",
    "ConversationSession",
    "IntentClassifier",
    "EntityExtractor",
    "DialogueManager",
    "DialogueState",
]
