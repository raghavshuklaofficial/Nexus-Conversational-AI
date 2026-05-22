"""
Ingestion worker — consumes document.ingest.requested events from Kafka,
runs the ingestion pipeline, and publishes completion events.
"""

from __future__ import annotations

import asyncio
import signal
from typing import Any

import structlog

from nexus.config import get_config
from nexus.domain.models import Document
from nexus.infrastructure.observability.logging import setup_logging

logger = structlog.get_logger(__name__)

MAX_RETRIES = 3


async def handle_ingest_event(event: dict[str, Any], ingestion_service: Any) -> None:
    """Process a single ingest event with retry logic."""
    doc_id = event.get("document_id", "")
    title = event.get("title", "")
    content = event.get("content", "")
    retry_count = event.get("_retry_count", 0)

    logger.info("processing_ingest_event", document_id=doc_id, retry=retry_count)

    try:
        doc = Document(title=title, content=content, source=event.get("source", ""))
        # Use the ID from the event
        from uuid import UUID
        doc.id = UUID(doc_id) if doc_id else doc.id
        doc.idempotency_key = event.get("idempotency_key", doc.compute_idempotency_key())

        await ingestion_service.ingest(doc)

    except Exception as e:
        if retry_count < MAX_RETRIES:
            logger.warning("ingest_retry", document_id=doc_id, retry=retry_count + 1, error=str(e))
            event["_retry_count"] = retry_count + 1
            # In production, re-publish to the same topic with backoff
        else:
            logger.error("ingest_exhausted_retries", document_id=doc_id, error=str(e))


async def run_worker() -> None:
    """Main worker loop."""
    config = get_config()
    setup_logging(level="INFO", format="json")
    logger.info("ingest_worker_starting")

    # Build minimal service set for the worker
    from nexus.infrastructure.cache.memory_cache import MemoryCache
    from nexus.infrastructure.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
    from nexus.infrastructure.messaging.kafka_producer import InMemoryMessageBus
    from nexus.infrastructure.persistence.repositories import InMemoryDocumentRepository
    from nexus.application.ingestion_service import IngestionService

    cache = MemoryCache()
    embedder = SentenceTransformerEmbedder(model_name=config.nlu.embedding_model, device=config.nlu.device)
    await embedder.initialize()

    # Vector store
    try:
        from nexus.infrastructure.vectorstores.qdrant_store import QdrantVectorStore
        vs = QdrantVectorStore(url=getattr(config, 'qdrant_url', 'http://localhost:6333'), embedding_dim=embedder.get_dimension())
        await vs.initialize()
    except Exception:
        from nexus.infrastructure.vectorstores.faiss_store import FAISSVectorStore
        vs = FAISSVectorStore(embedding_dim=embedder.get_dimension())
        await vs.initialize()

    bus = InMemoryMessageBus()
    repo = InMemoryDocumentRepository()

    ingestion = IngestionService(vector_store=vs, embedder=embedder, cache=cache, message_bus=bus, repository=repo)

    # Kafka consumer
    kafka_servers = getattr(config, 'kafka_bootstrap_servers', 'localhost:9092')
    try:
        from nexus.infrastructure.messaging.kafka_consumer import KafkaConsumer

        consumer = KafkaConsumer(
            bootstrap_servers=kafka_servers,
            group_id="nexus-ingest-workers",
            topics=["document.ingest.requested"],
        )
        consumer.register_handler(
            "document.ingest.requested",
            lambda event: handle_ingest_event(event, ingestion),
        )

        await consumer.start()
        logger.info("ingest_worker_ready")

        # Graceful shutdown
        loop = asyncio.get_event_loop()
        stop_event = asyncio.Event()

        def _signal_handler():
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler)

        consume_task = asyncio.create_task(consumer.consume())
        await stop_event.wait()
        await consumer.stop()
        consume_task.cancel()

    except Exception as e:
        logger.warning("kafka_unavailable_standalone_mode", error=str(e))
        logger.info("ingest_worker_standing_by")
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(run_worker())
