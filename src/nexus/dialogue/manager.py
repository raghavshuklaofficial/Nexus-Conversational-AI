"""
Dialogue manager - picks the right handler or template response
based on the current dialogue state.
"""

from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Any

import structlog

from nexus.config import DialogueConfig
from nexus.dialogue.state import DialogueState
from nexus.dialogue.handlers import HandlerRegistry, create_default_registry

logger = structlog.get_logger(__name__)


class DialogueManager:
    """
    Coordinates response generation using intent handlers and
    template responses. Handles multi-turn context and sentiment adaptation.
    """
    
    def __init__(
        self,
        config: DialogueConfig,
        embedding_engine: Any = None,
    ) -> None:
        self.config = config
        self._embedding_engine = embedding_engine
        self._handler_registry = create_default_registry()
        self._response_templates: dict[str, list[str]] = {}
        self._loaded = False
    
    async def load(self) -> None:
        """Load response templates and initialize components."""
        if self._loaded:
            return
        
        logger.info("loading_dialogue_manager")
        
        # Load response templates
        await self._load_response_templates()
        
        self._loaded = True
        logger.info("dialogue_manager_loaded")
    
    async def _load_response_templates(self) -> None:
        """Load response templates from data."""
        from nexus.data.intents import get_intent_responses
        
        self._response_templates = get_intent_responses()
    
    async def generate_response(
        self,
        state: DialogueState,
    ) -> dict[str, Any]:
        """Generate a response for the current dialogue state."""
        if not self._loaded:
            raise RuntimeError("Manager not loaded. Call load() first.")
        
        intent_name = state.intent.name
        
        # 1. Try a dedicated handler (greeting, goodbye, help, etc.)
        #    But NOT the default/fallback handler — we want to try templates first.
        handler = self._handler_registry.get_handler(state)
        
        if handler and handler.intent == intent_name:
            # Only use if it's a SPECIFIC handler for this intent, not the default fallback
            try:
                response = await handler.handle(state)
                response = self._adapt_response(response, state)
                return response
            except Exception as e:
                logger.error(
                    "handler_error",
                    handler=handler.__class__.__name__,
                    error=str(e),
                )
        
        # 2. Try template-based response from INTENT_DATABASE
        #    (this covers identity, joke, weather, thanks, booking, etc.)
        template_response = await self._generate_template_response(state)
        if template_response.get("type") != "fallback_default":
            return template_response
        
        # 3. Last resort: use the default fallback handler
        if handler:
            try:
                response = await handler.handle(state)
                response = self._adapt_response(response, state)
                return response
            except Exception as e:
                logger.error("fallback_handler_error", error=str(e))
        
        return template_response
    
    async def _generate_template_response(
        self,
        state: DialogueState,
    ) -> dict[str, Any]:
        """Generate response from templates."""
        intent_name = state.intent.name
        
        if intent_name in self._response_templates:
            templates = self._response_templates[intent_name]
            
            # pick a random response from the template list
            response_text = random.choice(templates)
            
            # Fill template slots with entities
            response_text = self._fill_template(response_text, state)
            
            return {
                "text": response_text,
                "type": "standard",
                "suggestions": self._get_suggestions(state),
            }
        
        # Default fallback — no template found for this intent
        return {
            "text": "I understand. How can I help you further?",
            "type": "fallback_default",
            "suggestions": ["Tell me more", "Help", "Something else"],
        }
    
    def _fill_template(self, template: str, state: DialogueState) -> str:
        """Fill template placeholders with entity values."""
        result = template
        
        for entity in state.entities:
            placeholder = f"{{{entity.type.lower()}}}"
            value = entity.value or entity.text
            result = result.replace(placeholder, value)
        
        return result
    
    def _adapt_response(
        self,
        response: dict[str, Any],
        state: DialogueState,
    ) -> dict[str, Any]:
        """Adapt response based on context and sentiment."""
        # Adapt to negative sentiment
        if state.is_negative_sentiment:
            if "text" in response:
                # Add empathetic prefix for negative sentiment
                empathy_phrases = [
                    "I understand this might be frustrating. ",
                    "I hear you, and I want to help. ",
                    "I appreciate your patience. ",
                ]
                response["text"] = random.choice(empathy_phrases) + response["text"]
        
        return response
    
    def _get_suggestions(self, state: DialogueState) -> list[str]:
        """Generate contextual suggestions."""
        base_suggestions = ["Tell me more", "Help", "Something else"]
        
        # Add intent-specific suggestions
        intent_suggestions = {
            "greeting": ["What can you do?", "I need help"],
            "booking": ["Check availability", "Cancel booking"],
            "support": ["Technical issue", "Account help"],
            "product_inquiry": ["Pricing", "Features"],
        }
        
        intent = state.intent.name
        if intent in intent_suggestions:
            return intent_suggestions[intent][:3]
        
        return base_suggestions
    
    def register_handler(self, handler: Any) -> None:
        """Register a custom intent handler."""
        self._handler_registry.register(handler)
    
    def __repr__(self) -> str:
        return f"DialogueManager(loaded={self._loaded})"
