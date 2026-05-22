"""
In-memory document repository.

For production you'd swap this for a PostgreSQL/SQLAlchemy implementation
behind the same DocumentRepositoryPort interface.
"""

from __future__ import annotations

from typing import Any

import structlog

from nexus.domain.models import Document, DocumentStatus
from nexus.domain.ports import DocumentRepositoryPort

logger = structlog.get_logger(__name__)


class InMemoryDocumentRepository(DocumentRepositoryPort):
    """Simple in-memory document store (suitable for single-process deployments)."""

    def __init__(self) -> None:
        self._store: dict[str, Document] = {}

    async def save(self, document: Any) -> None:
        doc: Document = document
        self._store[str(doc.id)] = doc
        logger.debug("document_saved", document_id=str(doc.id))

    async def get(self, document_id: str) -> Document | None:
        return self._store.get(document_id)

    async def update_status(
        self, document_id: str, status: str, error: str | None = None
    ) -> None:
        doc = self._store.get(document_id)
        if doc:
            doc.status = DocumentStatus(status)
            if error:
                doc.error_message = error
            logger.debug("document_status_updated", document_id=document_id, status=status)

    async def delete(self, document_id: str) -> None:
        self._store.pop(document_id, None)

    async def list_all(self, limit: int = 100, offset: int = 0) -> list[Document]:
        docs = list(self._store.values())
        return docs[offset:offset + limit]
