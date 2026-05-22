"""
Redis cache implementation.

Handles session state, response caching, embedding cache,
rate limiting, and distributed locks.
"""

from __future__ import annotations

import json
import hashlib
from typing import Any

import structlog

from nexus.domain.ports import CachePort

logger = structlog.get_logger(__name__)


class RedisCache(CachePort):
    """Production cache using Redis with structured key patterns."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0", default_ttl: int = 3600):
        self._redis_url = redis_url
        self._default_ttl = default_ttl
        self._client = None

    async def _get_client(self):
        if self._client is None:
            import redis.asyncio as aioredis
            self._client = aioredis.from_url(
                self._redis_url,
                decode_responses=True,
                max_connections=20,
            )
        return self._client

    async def get(self, key: str) -> Any | None:
        try:
            client = await self._get_client()
            value = await client.get(key)
            if value is not None:
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            return None
        except Exception as e:
            logger.warning("redis_get_error", key=key, error=str(e))
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        try:
            client = await self._get_client()
            serialized = json.dumps(value, default=str)
            await client.set(key, serialized, ex=ttl or self._default_ttl)
        except Exception as e:
            logger.warning("redis_set_error", key=key, error=str(e))

    async def delete(self, key: str) -> None:
        try:
            client = await self._get_client()
            await client.delete(key)
        except Exception as e:
            logger.warning("redis_delete_error", key=key, error=str(e))

    async def exists(self, key: str) -> bool:
        try:
            client = await self._get_client()
            return bool(await client.exists(key))
        except Exception:
            return False

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            return await client.ping()
        except Exception:
            return False

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    # ---- Session helpers ----

    async def get_session(self, session_id: str) -> dict[str, Any] | None:
        return await self.get(f"session:{session_id}")

    async def save_session(self, session_id: str, data: dict[str, Any], ttl: int = 1800) -> None:
        await self.set(f"session:{session_id}", data, ttl=ttl)

    async def delete_session(self, session_id: str) -> None:
        await self.delete(f"session:{session_id}")

    # ---- Response cache helpers ----

    @staticmethod
    def _cache_key(query: str, session_id: str = "", use_rag: bool = False) -> str:
        raw = f"{query.strip().lower()}:{session_id}:{use_rag}"
        h = hashlib.sha256(raw.encode()).hexdigest()[:16]
        return f"chat_cache:{h}"

    async def get_cached_response(self, query: str, session_id: str = "", use_rag: bool = False) -> dict | None:
        key = self._cache_key(query, session_id, use_rag)
        return await self.get(key)

    async def cache_response(
        self, query: str, response: dict, session_id: str = "", use_rag: bool = False, ttl: int = 300
    ) -> None:
        key = self._cache_key(query, session_id, use_rag)
        await self.set(key, response, ttl=ttl)

    # ---- Rate limiter ----

    async def check_rate_limit(self, client_id: str, limit: int = 60, window: int = 60) -> bool:
        """Returns True if the request is allowed."""
        try:
            client = await self._get_client()
            key = f"rate_limit:{client_id}"
            current = await client.incr(key)
            if current == 1:
                await client.expire(key, window)
            return current <= limit
        except Exception:
            return True  # fail open

    # ---- Distributed lock ----

    async def acquire_lock(self, resource: str, ttl: int = 300) -> bool:
        """Simple distributed lock using SET NX."""
        try:
            client = await self._get_client()
            key = f"ingest_lock:{resource}"
            return bool(await client.set(key, "1", ex=ttl, nx=True))
        except Exception:
            return False

    async def release_lock(self, resource: str) -> None:
        await self.delete(f"ingest_lock:{resource}")
