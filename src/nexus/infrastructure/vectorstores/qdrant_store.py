"""
Qdrant vector store — production backend.

Uses the official qdrant-client with gRPC for high-throughput upserts
and cosine similarity search.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import structlog

from nexus.domain.ports import VectorStorePort

logger = structlog.get_logger(__name__)


class QdrantVectorStore(VectorStorePort):
    """Qdrant-backed vector store."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "nexus_chunks",
        embedding_dim: int = 768,
        api_key: str | None = None,
    ):
        self._url = url
        self._collection_name = collection_name
        self._embedding_dim = embedding_dim
        self._api_key = api_key
        self._client = None

    async def initialize(self) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self._client = QdrantClient(url=self._url, api_key=self._api_key, timeout=30)

            # Create collection if it doesn't exist
            collections = self._client.get_collections().collections
            existing = [c.name for c in collections]

            if self._collection_name not in existing:
                self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=VectorParams(
                        size=self._embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(
                    "qdrant_collection_created",
                    collection=self._collection_name,
                    dim=self._embedding_dim,
                )
            else:
                logger.info("qdrant_collection_exists", collection=self._collection_name)

        except Exception as e:
            logger.error("qdrant_init_failed", error=str(e))
            raise

    async def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        texts: list[str],
    ) -> None:
        from qdrant_client.models import PointStruct

        points = []
        for id_, emb, meta, text in zip(ids, embeddings, metadatas, texts):
            payload = {**meta, "text": text}
            points.append(PointStruct(id=id_, vector=emb, payload=payload))

        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
            wait=True,
        )

        logger.info("qdrant_upsert", count=len(points))

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        qdrant_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            qdrant_filter = Filter(must=conditions)

        results = self._client.search(
            collection_name=self._collection_name,
            query_vector=query_embedding,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )

        return [
            {
                "id": str(hit.id),
                "score": hit.score,
                "text": hit.payload.get("text", ""),
                "document_id": hit.payload.get("document_id", ""),
                "chunk_index": hit.payload.get("chunk_index", 0),
                "metadata": {k: v for k, v in hit.payload.items() if k not in ("text",)},
            }
            for hit in results
        ]

    async def delete(self, ids: list[str]) -> None:
        from qdrant_client.models import PointIdsList

        self._client.delete(
            collection_name=self._collection_name,
            points_selector=PointIdsList(points=ids),
        )
        logger.info("qdrant_delete", count=len(ids))

    async def delete_by_metadata(self, filters: dict[str, Any]) -> int:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
        ]
        qdrant_filter = Filter(must=conditions)

        # Scroll to find matching point IDs, then delete
        results, _ = self._client.scroll(
            collection_name=self._collection_name,
            scroll_filter=qdrant_filter,
            limit=10_000,
            with_payload=False,
        )

        if results:
            ids = [str(p.id) for p in results]
            await self.delete(ids)
            return len(ids)
        return 0

    async def health_check(self) -> bool:
        try:
            if self._client is None:
                return False
            self._client.get_collections()
            return True
        except Exception:
            return False

    async def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None
