"""
In-memory cache for development and testing.
API-compatible with RedisCache so the application layer doesn't care.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any

from nexus.domain.ports import CachePort


class MemoryCache(CachePort):
    """Simple TTL-aware LRU cache for development."""

    def __init__(self, max_size: int = 10_000, default_ttl: int = 3600):
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl

    async def get(self, key: str) -> Any | None:
        if key in self._store:
            value, expires_at = self._store[key]
            if expires_at == 0 or time.time() < expires_at:
                self._store.move_to_end(key)
                return value
            else:
                del self._store[key]
        return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        ttl = ttl or self._default_ttl
        expires_at = time.time() + ttl if ttl > 0 else 0
        self._store[key] = (value, expires_at)
        self._store.move_to_end(key)
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def exists(self, key: str) -> bool:
        return await self.get(key) is not None

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        self._store.clear()

    # ---- Rate limiter (basic) ----
    async def check_rate_limit(self, client_id: str, limit: int = 60, window: int = 60) -> bool:
        key = f"rate_limit:{client_id}"
        entry = await self.get(key)
        if entry is None:
            await self.set(key, 1, ttl=window)
            return True
        count = int(entry)
        if count >= limit:
            return False
        await self.set(key, count + 1, ttl=window)
        return True

    # ---- Distributed lock (no-op for single process) ----
    async def acquire_lock(self, resource: str, ttl: int = 300) -> bool:
        key = f"ingest_lock:{resource}"
        if await self.exists(key):
            return False
        await self.set(key, "1", ttl=ttl)
        return True

    async def release_lock(self, resource: str) -> None:
        await self.delete(f"ingest_lock:{resource}")
