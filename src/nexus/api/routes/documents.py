"""Document routes — POST/GET/DELETE /documents"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks

from nexus.api.dependencies import get_ingestion_service, get_repository, get_message_bus
from nexus.api.schemas.documents import DocumentUploadRequest, DocumentResponse, DocumentDeleteResponse
from nexus.application.ingestion_service import IngestionService
from nexus.domain.models import Document
from nexus.infrastructure.persistence.repositories import InMemoryDocumentRepository

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("", response_model=DocumentResponse, status_code=201)
async def upload_document(
    request: DocumentUploadRequest,
    background_tasks: BackgroundTasks,
    ingestion: IngestionService = Depends(get_ingestion_service),
    repository: InMemoryDocumentRepository = Depends(get_repository),
    bus=Depends(get_message_bus),
):
    """Upload a document for ingestion. Processing happens asynchronously."""
    doc = Document(
        title=request.title,
        content=request.content,
        source=request.source,
        metadata=request.metadata,
    )
    doc.idempotency_key = doc.compute_idempotency_key()
    await repository.save(doc)

    # Publish ingest event
    await bus.publish("document.ingest.requested", {
        "event_type": "document.ingest.requested",
        "document_id": str(doc.id),
        "title": doc.title,
        "idempotency_key": doc.idempotency_key,
    })

    # Also kick off inline ingestion in background
    background_tasks.add_task(ingestion.ingest, doc)

    return DocumentResponse(
        document_id=str(doc.id),
        title=doc.title,
        status=doc.status.value,
        source=doc.source,
        created_at=doc.created_at.isoformat(),
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    repository: InMemoryDocumentRepository = Depends(get_repository),
):
    """Get document ingestion status."""
    doc = await repository.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentResponse(
        document_id=str(doc.id),
        title=doc.title,
        status=doc.status.value,
        chunk_count=doc.chunk_count,
        source=doc.source,
        error_message=doc.error_message,
        created_at=doc.created_at.isoformat(),
    )


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    document_id: str,
    ingestion: IngestionService = Depends(get_ingestion_service),
    repository: InMemoryDocumentRepository = Depends(get_repository),
):
    """Delete a document and its vector entries."""
    doc = await repository.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    await ingestion.delete_document(document_id)
    return DocumentDeleteResponse(document_id=document_id)
