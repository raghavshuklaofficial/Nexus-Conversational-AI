"""
Kafka producer — publishes events to Kafka topics.

Falls back to a no-op in-memory implementation when Kafka is unavailable,
so the application layer always works.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from nexus.domain.ports import MessageBusPort

logger = structlog.get_logger(__name__)


class KafkaProducer(MessageBusPort):
    """Async Kafka producer using aiokafka."""

    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        self._bootstrap_servers = bootstrap_servers
        self._producer = None

    async def initialize(self) -> None:
        try:
            from aiokafka import AIOKafkaProducer

            self._producer = AIOKafkaProducer(
                bootstrap_servers=self._bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode(),
                key_serializer=lambda k: k.encode() if k else None,
            )
            await self._producer.start()
            logger.info("kafka_producer_started", servers=self._bootstrap_servers)
        except Exception as e:
            logger.warning("kafka_producer_unavailable", error=str(e))
            self._producer = None

    async def publish(self, topic: str, event: dict[str, Any]) -> None:
        if self._producer is None:
            logger.debug("kafka_publish_skipped", topic=topic, reason="no producer")
            return

        key = event.get("idempotency_key") or event.get("event_id", "")
        try:
            await self._producer.send_and_wait(topic, value=event, key=key)
            logger.info("kafka_event_published", topic=topic, event_type=event.get("event_type"))
        except Exception as e:
            logger.error("kafka_publish_error", topic=topic, error=str(e))

    async def health_check(self) -> bool:
        return self._producer is not None

    async def close(self) -> None:
        if self._producer:
            await self._producer.stop()
            self._producer = None


class InMemoryMessageBus(MessageBusPort):
    """In-memory fallback for development/testing."""

    def __init__(self) -> None:
        self.published: list[tuple[str, dict[str, Any]]] = []

    async def publish(self, topic: str, event: dict[str, Any]) -> None:
        self.published.append((topic, event))
        logger.debug("inmemory_event_published", topic=topic, event_type=event.get("event_type"))

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        self.published.clear()
