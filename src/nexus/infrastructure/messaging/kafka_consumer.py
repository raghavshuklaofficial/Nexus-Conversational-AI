"""
Kafka consumer — consumes events from Kafka topics and dispatches to handlers.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Coroutine

import structlog

logger = structlog.get_logger(__name__)


class KafkaConsumer:
    """Async Kafka consumer using aiokafka."""

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        group_id: str = "nexus-workers",
        topics: list[str] | None = None,
    ):
        self._bootstrap_servers = bootstrap_servers
        self._group_id = group_id
        self._topics = topics or []
        self._consumer = None
        self._handlers: dict[str, Callable] = {}
        self._running = False

    def register_handler(
        self,
        event_type: str,
        handler: Callable[[dict[str, Any]], Coroutine],
    ) -> None:
        """Register a handler for a specific event type."""
        self._handlers[event_type] = handler
        logger.info("kafka_handler_registered", event_type=event_type)

    async def start(self) -> None:
        """Start consuming messages."""
        try:
            from aiokafka import AIOKafkaConsumer

            self._consumer = AIOKafkaConsumer(
                *self._topics,
                bootstrap_servers=self._bootstrap_servers,
                group_id=self._group_id,
                value_deserializer=lambda v: json.loads(v.decode()),
                auto_offset_reset="earliest",
                enable_auto_commit=False,
            )

            await self._consumer.start()
            self._running = True
            logger.info("kafka_consumer_started", topics=self._topics, group=self._group_id)

        except Exception as e:
            logger.error("kafka_consumer_start_failed", error=str(e))
            raise

    async def consume(self) -> None:
        """Main consume loop — dispatches events to registered handlers."""
        if not self._consumer:
            raise RuntimeError("Consumer not started")

        try:
            async for message in self._consumer:
                if not self._running:
                    break

                event = message.value
                event_type = event.get("event_type", "unknown")

                logger.info(
                    "kafka_event_received",
                    topic=message.topic,
                    event_type=event_type,
                    offset=message.offset,
                )

                handler = self._handlers.get(event_type)
                if handler:
                    try:
                        await handler(event)
                        await self._consumer.commit()
                    except Exception as e:
                        logger.error(
                            "kafka_handler_error",
                            event_type=event_type,
                            error=str(e),
                        )
                        # Don't commit — message will be retried
                else:
                    logger.warning("kafka_no_handler", event_type=event_type)
                    await self._consumer.commit()

        except asyncio.CancelledError:
            logger.info("kafka_consumer_cancelled")
        except Exception as e:
            logger.error("kafka_consume_error", error=str(e))

    async def stop(self) -> None:
        """Stop the consumer gracefully."""
        self._running = False
        if self._consumer:
            await self._consumer.stop()
            self._consumer = None
            logger.info("kafka_consumer_stopped")
