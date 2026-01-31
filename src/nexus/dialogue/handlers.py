"""
Intent Handlers
===============

Pluggable intent handling system for customizable response generation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Coroutine

import structlog

from nexus.dialogue.state import DialogueState

logger = structlog.get_logger(__name__)


class IntentHandler(ABC):
    """
    Base class for intent handlers.
    
    Implement custom handlers for specific intents to provide
    domain-specific response logic.
    
    Example:
        >>> class BookingHandler(IntentHandler):
        ...     intent = "booking"
        ...     
        ...     async def handle(self, state: DialogueState) -> dict:
        ...         # Custom booking logic
        ...         return {"text": "Let me help you with booking..."}
    """
    
    intent: str = ""
    priority: int = 0
    
    @abstractmethod
    async def handle(self, state: DialogueState) -> dict[str, Any]:
        """
        Handle the intent and generate response data.
        
        Args:
            state: Current dialogue state
        
        Returns:
            dict: Response data with 'text' and optional fields
        """
        pass
    
    def can_handle(self, state: DialogueState) -> bool:
        """
        Check if this handler can process the given state.
        
        Override for conditional handling beyond intent matching.
        
        Args:
            state: Current dialogue state
        
        Returns:
            bool: True if handler can process the state
        """
        return state.intent.name == self.intent


class HandlerRegistry:
    """
    Registry for intent handlers.
    
    Manages registration and lookup of handlers for different intents.
    Supports priority-based handler selection and fallback handling.
    """
    
    def __init__(self) -> None:
        """Initialize the handler registry."""
        self._handlers: dict[str, list[IntentHandler]] = {}
        self._default_handler: IntentHandler | None = None
    
    def register(self, handler: IntentHandler) -> None:
        """
        Register an intent handler.
        
        Args:
            handler: Handler instance to register
        """
        intent = handler.intent
        
        if intent not in self._handlers:
            self._handlers[intent] = []
        
        self._handlers[intent].append(handler)
        
        # Sort by priority (highest first)
        self._handlers[intent].sort(key=lambda h: h.priority, reverse=True)
        
        logger.debug(
            "handler_registered",
            intent=intent,
            handler=handler.__class__.__name__,
        )
    
    def set_default(self, handler: IntentHandler) -> None:
        """Set the default fallback handler."""
        self._default_handler = handler
    
    def get_handler(self, state: DialogueState) -> IntentHandler | None:
        """
        Get the appropriate handler for the given state.
        
        Args:
            state: Current dialogue state
        
        Returns:
            IntentHandler: Matching handler or default
        """
        intent = state.intent.name
        
        # Check registered handlers
        if intent in self._handlers:
            for handler in self._handlers[intent]:
                if handler.can_handle(state):
                    return handler
        
        # Return default if available
        return self._default_handler
    
    def handler(
        self,
        intent: str,
        priority: int = 0,
    ) -> Callable[[type[IntentHandler]], type[IntentHandler]]:
        """
        Decorator for registering handlers.
        
        Example:
            >>> @registry.handler("greeting")
            ... class GreetingHandler(IntentHandler):
            ...     async def handle(self, state):
            ...         return {"text": "Hello!"}
        """
        def decorator(cls: type[IntentHandler]) -> type[IntentHandler]:
            cls.intent = intent
            cls.priority = priority
            self.register(cls())
            return cls
        return decorator


# Built-in handlers

class GreetingHandler(IntentHandler):
    """Handler for greeting intents."""
    
    intent = "greeting"
    
    async def handle(self, state: DialogueState) -> dict[str, Any]:
        """Generate greeting response."""
        hour = datetime.now().hour
        
        if hour < 12:
            time_greeting = "Good morning"
        elif hour < 17:
            time_greeting = "Good afternoon"
        else:
            time_greeting = "Good evening"
        
        # Personalize if we have history
        if state.session and state.session.turn_count > 0:
            return {
                "text": f"Welcome back! How can I assist you today?",
                "type": "standard",
                "suggestions": ["Ask a question", "Get help", "Learn more"],
            }
        
        return {
            "text": f"{time_greeting}! I'm Nexus, your AI assistant. How can I help you today?",
            "type": "standard",
            "suggestions": ["What can you do?", "Tell me about yourself", "Help me with something"],
        }


class GoodbyeHandler(IntentHandler):
    """Handler for goodbye intents."""
    
    intent = "goodbye"
    
    async def handle(self, state: DialogueState) -> dict[str, Any]:
        """Generate farewell response."""
        if state.session and state.session.turn_count > 5:
            return {
                "text": "It was great chatting with you! Feel free to return anytime. Take care! 👋",
                "type": "standard",
            }
        
        return {
            "text": "Goodbye! Have a wonderful day! 👋",
            "type": "standard",
        }


class HelpHandler(IntentHandler):
    """Handler for help requests."""
    
    intent = "help"
    
    async def handle(self, state: DialogueState) -> dict[str, Any]:
        """Generate help response."""
        return {
            "text": (
                "I'm here to help! Here's what I can assist you with:\n\n"
                "• **General Questions** - Ask me anything and I'll do my best to help\n"
                "• **Information Lookup** - I can provide information on various topics\n"
                "• **Recommendations** - Get suggestions for movies, music, and more\n"
                "• **Scheduling** - Help with appointments and bookings\n"
                "• **Support** - Technical assistance and troubleshooting\n\n"
                "Just type your question or request, and I'll guide you through it!"
            ),
            "type": "standard",
            "suggestions": ["Ask a question", "Make a booking", "Get support"],
        }


class FallbackHandler(IntentHandler):
    """Handler for unrecognized intents."""
    
    intent = "fallback"
    
    async def handle(self, state: DialogueState) -> dict[str, Any]:
        """Generate fallback response."""
        confidence = state.intent.confidence
        
        if confidence < 0.2:
            return {
                "text": (
                    "I'm not quite sure I understood that. Could you please rephrase "
                    "your question or provide more details?"
                ),
                "type": "clarification",
                "suggestions": ["Help", "What can you do?"],
            }
        
        return {
            "text": (
                "I'm not entirely sure how to help with that specific request, but I'd love to try! "
                "Could you provide a bit more context or ask in a different way?"
            ),
            "type": "fallback",
            "suggestions": ["Try something else", "Get help"],
        }


# Create default registry with built-in handlers
def create_default_registry() -> HandlerRegistry:
    """Create a handler registry with default handlers."""
    registry = HandlerRegistry()
    
    registry.register(GreetingHandler())
    registry.register(GoodbyeHandler())
    registry.register(HelpHandler())
    
    fallback = FallbackHandler()
    registry.register(fallback)
    registry.set_default(fallback)
    
    return registry
