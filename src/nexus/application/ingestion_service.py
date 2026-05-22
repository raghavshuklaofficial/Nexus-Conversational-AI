"""
Ingestion Service — handles document chunking, embedding, and indexing.
"""

from __future__ import annotations

import re
import time
from typing import Any
from uuid import uuid4

import structlog

from nexus.domain.models import Chunk, Document, DocumentStatus
from nexus.domain.ports import CachePort, EmbeddingPort, MessageBusPort, VectorStorePort, DocumentRepositoryPort
from nexus.infrastructure.observability.metrics import INGESTION_TOTAL, INGESTION_CHUNKS

logger = structlog.get_logger(__name__)

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64


class IngestionService:
    """Chunks, embeds, and indexes documents."""

    def __init__(
        self,
        vector_store: VectorStorePort,
        embedder: EmbeddingPort,
        cache: CachePort,
        message_bus: MessageBusPort,
        repository: DocumentRepositoryPort,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self._vector_store = vector_store
        self._embedder = embedder
        self._cache = cache
        self._bus = message_bus
        self._repo = repository
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, doc_id: str) -> list[Chunk]:
        """Split text into overlapping chunks at sentence boundaries."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunks: list[Chunk] = []
        current, current_len, idx = "", 0, 0

        for sentence in sentences:
            if current_len + len(sentence) > self._chunk_size and current:
                chunks.append(Chunk(
                    document_id=doc_id, content=current.strip(),
                    index=idx, token_count=len(current.split()),
                ))
                # Keep overlap
                words = current.split()
                overlap_words = words[-self._chunk_overlap:] if len(words) > self._chunk_overlap else words
                current = " ".join(overlap_words) + " " + sentence
                current_len = len(current)
                idx += 1
            else:
                current += " " + sentence
                current_len += len(sentence) + 1

        if current.strip():
            chunks.append(Chunk(
                document_id=doc_id, content=current.strip(),
                index=idx, token_count=len(current.split()),
            ))
        return chunks

    async def ingest(self, document: Document) -> Document:
        """Full ingestion pipeline: chunk → embed → index."""
        doc_id = str(document.id)

        # Idempotency check
        locked = await self._cache.acquire_lock(doc_id, ttl=300)
        if not locked:
            logger.warning("ingestion_already_running", document_id=doc_id)
            return document

        try:
            # Update status → chunking
            document.status = DocumentStatus.CHUNKING
            await self._repo.save(document)

            chunks = self.chunk_text(document.content, doc_id)
            document.chunk_count = len(chunks)
            INGESTION_CHUNKS.inc(len(chunks))

            # Embed
            document.status = DocumentStatus.EMBEDDING
            await self._repo.update_status(doc_id, DocumentStatus.EMBEDDING.value)

            texts = [c.content for c in chunks]
            embeddings = await self._embedder.embed_texts(texts)

            # Index
            document.status = DocumentStatus.INDEXING
            await self._repo.update_status(doc_id, DocumentStatus.INDEXING.value)

            ids = [str(c.id) for c in chunks]
            metadatas = [
                {"document_id": doc_id, "chunk_index": c.index,
                 "source": document.source, "title": document.title}
                for c in chunks
            ]
            await self._vector_store.upsert(ids, embeddings, metadatas, texts)

            # Done
            document.status = DocumentStatus.INDEXED
            await self._repo.update_status(doc_id, DocumentStatus.INDEXED.value)
            INGESTION_TOTAL.labels(status="success").inc()

            # Publish event
            await self._bus.publish("document.indexed", {
                "event_type": "document.indexed",
                "document_id": doc_id, "chunk_count": len(chunks),
            })

            logger.info("document_ingested", document_id=doc_id, chunks=len(chunks))
            return document

        except Exception as e:
            document.status = DocumentStatus.FAILED
            document.error_message = str(e)
            await self._repo.update_status(doc_id, DocumentStatus.FAILED.value, error=str(e))
            INGESTION_TOTAL.labels(status="failure").inc()

            await self._bus.publish("dead.letter", {
                "event_type": "dead.letter", "original_event_type": "document.ingest",
                "error_message": str(e), "document_id": doc_id,
            })
            logger.error("ingestion_failed", document_id=doc_id, error=str(e))
            raise

        finally:
            await self._cache.release_lock(doc_id)

    async def delete_document(self, document_id: str) -> None:
        """Delete document and its vectors."""
        await self._vector_store.delete_by_metadata({"document_id": document_id})
        await self._repo.delete(document_id)
        logger.info("document_deleted", document_id=document_id)
