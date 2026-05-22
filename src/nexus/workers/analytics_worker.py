"""
Analytics worker — consumes chat.analytics events for processing.
"""

from __future__ import annotations

import asyncio
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


async def handle_analytics_event(event: dict[str, Any]) -> None:
    """Process a chat analytics event."""
    logger.info(
        "analytics_event",
        session_id=event.get("session_id"),
        intent=event.get("intent"),
        latency_ms=event.get("latency_ms"),
        cache_hit=event.get("cache_hit"),
        rag_used=event.get("rag_used"),
    )


async def run_worker() -> None:
    """Main analytics worker loop."""
    from nexus.infrastructure.observability.logging import setup_logging
    setup_logging(level="INFO", format="json")
    logger.info("analytics_worker_starting")

    try:
        from nexus.infrastructure.messaging.kafka_consumer import KafkaConsumer
        consumer = KafkaConsumer(
            bootstrap_servers="localhost:9092",
            group_id="nexus-analytics-workers",
            topics=["chat.analytics"],
        )
        consumer.register_handler("chat.analytics", handle_analytics_event)
        await consumer.start()
        await consumer.consume()
    except Exception as e:
        logger.warning("analytics_worker_kafka_unavailable", error=str(e))
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(run_worker())
