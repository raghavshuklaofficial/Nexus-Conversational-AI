"""
RAG Service — Retrieval-Augmented Generation pipeline.

Steps:
1. Generate query embedding
2. Check semantic cache
3. Retrieve top-k chunks from vector store
4. Rerank by cosine similarity
5. Build prompt with system instruction + context + history + query
6. Generate answer via LLM provider
7. Extract citations
8. Cache response
9. Emit metrics
"""

from __future__ import annotations

import re
import time
from typing import Any

import structlog

from nexus.domain.models import ChatResponse, Citation, LatencyBreakdown, RetrievedChunk
from nexus.domain.ports import CachePort, EmbeddingPort, LLMProviderPort, VectorStorePort
from nexus.infrastructure.observability.metrics import (
    CACHE_HITS, CACHE_MISSES, MODEL_LATENCY, RETRIEVAL_LATENCY,
)

logger = structlog.get_logger(__name__)

# Prompt injection patterns
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?above",
    r"you\s+are\s+now\s+",
    r"system\s*:\s*",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)

SYSTEM_PROMPT = (
    "You are Nexus, a helpful AI assistant. Answer the user's question using the "
    "provided context. If the context doesn't contain relevant information, say so "
    "honestly. Always cite specific sources when using retrieved information. "
    "Be concise and accurate."
)


class RAGService:
    """Async, observable RAG pipeline."""

    def __init__(
        self,
        vector_store: VectorStorePort,
        embedder: EmbeddingPort,
        llm: LLMProviderPort,
        cache: CachePort,
    ):
        self._vector_store = vector_store
        self._embedder = embedder
        self._llm = llm
        self._cache = cache

    @staticmethod
    def sanitize_query(query: str) -> str:
        """Basic prompt-injection mitigation."""
        if _INJECTION_RE.search(query):
            logger.warning("prompt_injection_detected", query=query[:100])
            return re.sub(_INJECTION_RE, "[FILTERED]", query)
        return query

    async def retrieve_and_generate(
        self,
        query: str,
        session_id: str = "",
        conversation_history: str = "",
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> tuple[str, list[RetrievedChunk], list[Citation], LatencyBreakdown, bool]:
        """Run the full RAG pipeline. Returns (answer, chunks, citations, latency, cache_hit)."""
        latency = LatencyBreakdown()
        query = self.sanitize_query(query)

        # 1. Check cache
        t0 = time.time()
        cache_key = f"rag:{hash(query + session_id)}"
        cached = await self._cache.get(cache_key)
        latency.cache_lookup_ms = (time.time() - t0) * 1000

        if cached:
            CACHE_HITS.labels(cache_type="rag").inc()
            return (
                cached["answer"],
                [RetrievedChunk(**c) for c in cached.get("chunks", [])],
                [Citation(**c) for c in cached.get("citations", [])],
                latency,
                True,
            )
        CACHE_MISSES.labels(cache_type="rag").inc()

        # 2. Embed query
        t1 = time.time()
        query_embedding = await self._embedder.embed_query(query)
        embed_time = time.time() - t1
        MODEL_LATENCY.labels(model_type="embedding").observe(embed_time)

        # 3. Retrieve
        t2 = time.time()
        raw_results = await self._vector_store.search(
            query_embedding=query_embedding, top_k=top_k, filters=filters,
        )
        latency.retrieval_ms = (time.time() - t2) * 1000
        RETRIEVAL_LATENCY.observe(latency.retrieval_ms / 1000)

        # 4. Build chunks and citations
        chunks = [
            RetrievedChunk(
                chunk_id=r["id"], document_id=r.get("document_id", ""),
                content=r.get("text", ""), score=r.get("score", 0.0),
                metadata=r.get("metadata", {}),
            ) for r in raw_results
        ]
        citations = [
            Citation(
                document_id=c.document_id, chunk_id=c.chunk_id, score=c.score,
                source=c.metadata.get("source", ""), title=c.metadata.get("title", ""),
                snippet=c.content[:200],
            ) for c in chunks
        ]

        # 5. Build prompt
        context = "\n\n".join(
            f"[Source {i+1}]: {c.content}" for i, c in enumerate(chunks)
        ) if chunks else "No relevant context found."

        prompt = self._build_prompt(query, context, conversation_history)

        # 6. Generate
        t3 = time.time()
        answer = await self._llm.generate(prompt)
        latency.generation_ms = (time.time() - t3) * 1000
        MODEL_LATENCY.labels(model_type="llm").observe(latency.generation_ms / 1000)

        latency.total_ms = (time.time() - t0) * 1000

        # 7. Cache
        await self._cache.set(cache_key, {
            "answer": answer,
            "chunks": [c.model_dump() for c in chunks],
            "citations": [c.model_dump() for c in citations],
        }, ttl=300)

        return answer, chunks, citations, latency, False

    def _build_prompt(self, query: str, context: str, history: str) -> str:
        parts = [f"System: {SYSTEM_PROMPT}"]
        if context:
            parts.append(f"\nRetrieved Context:\n{context}")
        if history:
            parts.append(f"\nConversation History:\n{history}")
        parts.append(f"\nUser: {query}\nAssistant:")
        return "\n".join(parts)
