"""
Nexus Conversational AI

Transformer-based chatbot with intent classification, NER,
sentiment analysis and multi-turn dialogue support.
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
]
