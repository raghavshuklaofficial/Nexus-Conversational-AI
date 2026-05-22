"""
Session Service — Redis-backed session management with in-memory fallback.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import structlog

from nexus.domain.models import ConversationMemory, Message, MessageRole
from nexus.infrastructure.cache.redis_cache import RedisCache
from nexus.infrastructure.cache.memory_cache import MemoryCache
from nexus.infrastructure.observability.metrics import ACTIVE_SESSIONS

logger = structlog.get_logger(__name__)


class SessionService:
    """Manages conversation sessions backed by cache."""

    def __init__(self, cache: RedisCache | MemoryCache, session_ttl: int = 1800):
        self._cache = cache
        self._session_ttl = session_ttl

    async def create_session(self) -> ConversationMemory:
        session_id = str(uuid4())
        memory = ConversationMemory(session_id=session_id)
        await self._save(memory)
        ACTIVE_SESSIONS.inc()
        logger.info("session_created", session_id=session_id)
        return memory

    async def get_session(self, session_id: str) -> ConversationMemory | None:
        data = await self._cache.get(f"session:{session_id}")
        if data is None:
            return None
        try:
            return ConversationMemory.model_validate(data)
        except Exception:
            return None

    async def add_turn(self, session_id: str, user_text: str, assistant_text: str) -> None:
        memory = await self.get_session(session_id)
        if memory is None:
            memory = ConversationMemory(session_id=session_id)
        
        memory.add_turn(user_text, assistant_text)
        self.prune_memory(memory)
        await self._save(memory)
        logger.debug(
            "memory_turn_added", 
            session_id=session_id, 
            message_count=len(memory.messages)
        )

    def prune_memory(self, memory: ConversationMemory, max_messages: int = 40) -> None:
        """Keep only the most recent messages to prevent token overflow."""
        if len(memory.messages) > max_messages:
            pruned_count = len(memory.messages) - max_messages
            memory.messages = memory.messages[-max_messages:]
            logger.info("memory_pruned", session_id=memory.session_id, removed_messages=pruned_count)

    async def delete_session(self, session_id: str) -> bool:
        exists = await self._cache.exists(f"session:{session_id}")
        if exists:
            await self._cache.delete(f"session:{session_id}")
            ACTIVE_SESSIONS.dec()
            logger.info("session_deleted", session_id=session_id)
            return True
        return False

    async def _save(self, memory: ConversationMemory) -> None:
        data = memory.model_dump(mode="json")
        await self._cache.set(f"session:{memory.session_id}", data, ttl=self._session_ttl)
        logger.debug("session_saved_to_cache", session_id=memory.session_id, bytes=len(json.dumps(data)))
