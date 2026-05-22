"""
FAISS vector store — lightweight local fallback for development.

Same interface as QdrantVectorStore so the application layer
doesn't need to know which backend is in use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from nexus.domain.ports import VectorStorePort

logger = structlog.get_logger(__name__)


class FAISSVectorStore(VectorStorePort):
    """FAISS-backed vector store for local development."""

    def __init__(self, embedding_dim: int = 768, index_path: str | None = None):
        self._embedding_dim = embedding_dim
        self._index_path = index_path
        self._index = None
        self._id_map: dict[int, str] = {}         # faiss internal → external id
        self._payloads: dict[str, dict[str, Any]] = {}  # external id → payload
        self._next_internal_id = 0

    async def initialize(self) -> None:
        try:
            import faiss

            self._index = faiss.IndexFlatIP(self._embedding_dim)  # inner product (we normalise to get cosine)
            logger.info("faiss_index_created", dim=self._embedding_dim)

            # Load persisted index if available
            if self._index_path and Path(self._index_path).exists():
                self._index = faiss.read_index(self._index_path)
                meta_path = Path(self._index_path).with_suffix(".meta.json")
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    self._id_map = {int(k): v for k, v in meta.get("id_map", {}).items()}
                    self._payloads = meta.get("payloads", {})
                    self._next_internal_id = max(self._id_map.keys(), default=-1) + 1
                logger.info("faiss_index_loaded", path=self._index_path)

        except ImportError:
            logger.warning("faiss_not_installed", message="pip install faiss-cpu")
            raise
        except Exception as e:
            logger.error("faiss_init_failed", error=str(e))
            raise

    async def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        texts: list[str],
    ) -> None:
        vectors = np.array(embeddings, dtype=np.float32)
        # L2-normalise for cosine similarity via inner product
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms

        # Remove existing ids first (simple re-index)
        # FAISS flat index doesn't support true upsert, so we just append
        for ext_id, emb, meta, text in zip(ids, vectors, metadatas, texts):
            internal_id = self._next_internal_id
            self._id_map[internal_id] = ext_id
            self._payloads[ext_id] = {**meta, "text": text}
            self._next_internal_id += 1

        self._index.add(vectors)
        logger.info("faiss_upsert", count=len(ids))

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query = np.array([query_embedding], dtype=np.float32)
        norms = np.linalg.norm(query, axis=1, keepdims=True)
        norms[norms == 0] = 1
        query = query / norms

        # Search more than top_k to allow post-filtering
        search_k = min(top_k * 3, self._index.ntotal) if self._index.ntotal > 0 else 0
        if search_k == 0:
            return []

        scores, indices = self._index.search(query, search_k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            ext_id = self._id_map.get(int(idx), "")
            if not ext_id:
                continue
            payload = self._payloads.get(ext_id, {})

            # Apply metadata filters
            if filters:
                if not all(payload.get(k) == v for k, v in filters.items()):
                    continue

            results.append({
                "id": ext_id,
                "score": float(score),
                "text": payload.get("text", ""),
                "document_id": payload.get("document_id", ""),
                "chunk_index": payload.get("chunk_index", 0),
                "metadata": payload,
            })

            if len(results) >= top_k:
                break

        return results

    async def delete(self, ids: list[str]) -> None:
        # FAISS flat index doesn't support deletion — we mark as removed
        ids_set = set(ids)
        for internal_id, ext_id in list(self._id_map.items()):
            if ext_id in ids_set:
                del self._id_map[internal_id]
                self._payloads.pop(ext_id, None)
        logger.info("faiss_delete_marked", count=len(ids))

    async def delete_by_metadata(self, filters: dict[str, Any]) -> int:
        to_delete = []
        for ext_id, payload in self._payloads.items():
            if all(payload.get(k) == v for k, v in filters.items()):
                to_delete.append(ext_id)
        await self.delete(to_delete)
        return len(to_delete)

    async def health_check(self) -> bool:
        return self._index is not None

    async def close(self) -> None:
        # Persist if path configured
        if self._index_path and self._index is not None:
            import faiss
            faiss.write_index(self._index, self._index_path)
            meta_path = Path(self._index_path).with_suffix(".meta.json")
            meta_path.write_text(json.dumps({
                "id_map": {str(k): v for k, v in self._id_map.items()},
                "payloads": self._payloads,
            }))
        self._index = None
