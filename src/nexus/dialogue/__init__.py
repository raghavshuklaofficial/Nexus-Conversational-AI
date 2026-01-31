"""
Dialogue Management Module
==========================

Sophisticated dialogue state tracking and response generation.
"""

from nexus.dialogue.manager import DialogueManager
from nexus.dialogue.state import DialogueState
from nexus.dialogue.handlers import IntentHandler, HandlerRegistry

__all__ = [
    "DialogueManager",
    "DialogueState",
    "IntentHandler",
    "HandlerRegistry",
]
