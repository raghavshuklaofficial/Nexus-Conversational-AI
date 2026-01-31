"""
Data Module
===========

Intent patterns, responses, and training data management.
"""

from nexus.data.intents import (
    get_intent_patterns,
    get_intent_responses,
    IntentData,
)

__all__ = [
    "get_intent_patterns",
    "get_intent_responses",
    "IntentData",
]
