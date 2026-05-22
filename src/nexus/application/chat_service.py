"""
Chat Service — top-level orchestrator for chat requests.

Coordinates NLU, RAG (optional), dialogue management, session memory,
caching, and analytics event emission.
"""

from __future__ import annotations

import time
from typing import Any

import structlog

from nexus.application.nlu_service import NLUService
from nexus.application.rag_service import RAGService
from nexus.application.session_service import SessionService
from nexus.domain.models import ChatResponse, LatencyBreakdown
from nexus.domain.ports import CachePort, MessageBusPort
from nexus.dialogue.manager import DialogueManager
from nexus.dialogue.state import DialogueState
from nexus.infrastructure.observability.metrics import CACHE_HITS, CACHE_MISSES

logger = structlog.get_logger(__name__)


class ChatService:
    """Orchestrates a chat turn — NLU + optional RAG + response generation."""

    def __init__(
        self,
        nlu: NLUService,
        rag: RAGService,
        sessions: SessionService,
        dialogue_manager: DialogueManager,
        cache: CachePort,
        message_bus: MessageBusPort,
    ):
        self._nlu = nlu
        self._rag = rag
        self._sessions = sessions
        self._dialogue = dialogue_manager
        self._cache = cache
        self._bus = message_bus

    async def chat(
        self,
        text: str,
        session_id: str | None = None,
        use_rag: bool = False,
        top_k: int = 5,
        request_id: str = "",
    ) -> ChatResponse:
        """Process a chat message end-to-end."""
        total_start = time.time()
        latency = LatencyBreakdown()

        # Normalize
        text = " ".join(text.split()).strip()
        if not text:
            return ChatResponse(answer="Please provide a message.", session_id=session_id or "")

        # Session
        if session_id:
            memory = await self._sessions.get_session(session_id)
            if memory is None:
                memory = await self._sessions.create_session()
                session_id = memory.session_id
        else:
            memory = await self._sessions.create_session()
            session_id = memory.session_id

        history = memory.format_history(n_turns=3) if memory else ""
        cache_key_raw = f"{text}:{history}:{use_rag}"
        import hashlib
        cache_key = f"chat_cache:{hashlib.sha256(cache_key_raw.encode()).hexdigest()[:16]}"

        # Check response cache
        cached = await self._cache.get(cache_key)
        if cached and isinstance(cached, dict):
            CACHE_HITS.labels(cache_type="chat").inc()
            cached["cache_hit"] = True
            cached["request_id"] = request_id
            return ChatResponse(**cached)
        CACHE_MISSES.labels(cache_type="chat").inc()

        # NLU
        nlu_result = await self._nlu.analyze(text)
        latency.nlu_ms = nlu_result.latency_ms

        intent_name = nlu_result.intent.name

        # RAG or dialogue
        answer = ""
        chunks, citations = [], []
        cache_hit = False
        model_name = ""

        if use_rag:
            answer, chunks, citations, rag_latency, cache_hit = await self._rag.retrieve_and_generate(
                query=text, session_id=session_id, conversation_history=history, top_k=top_k,
            )
            latency.retrieval_ms = rag_latency.retrieval_ms
            latency.generation_ms = rag_latency.generation_ms
            latency.rerank_ms = rag_latency.rerank_ms
            model_name = "rag-pipeline"
        else:
            # Use dialogue manager for known intents
            state = DialogueState(
                user_input=text,
                intent=nlu_result.intent,
                entities=nlu_result.entities,
                sentiment=(nlu_result.sentiment, nlu_result.sentiment_score),
            )
            gen_start = time.time()
            response_data = await self._dialogue.generate_response(state)
            latency.generation_ms = (time.time() - gen_start) * 1000
            answer = response_data.get("text", "")
            model_name = "dialogue-manager"

            # If the dialogue manager returned a fallback, use the LLM to
            # actually answer the question — this is what makes it a real chatbot.
            is_fallback = (
                response_data.get("type") in ("fallback", "fallback_default", "clarification")
                or nlu_result.intent.name == "fallback"
            )
            if is_fallback and self._rag._llm is not None:
                logger.info("llm_fallback", query=text[:80])
                prompt = (
                    "You are Nexus, a helpful and friendly AI assistant. "
                    "Answer the user's question directly and concisely.\n"
                )
                if history:
                    prompt += f"\nConversation History:\n{history}\n"
                prompt += f"\nUser: {text}\nAssistant:"
                gen_start = time.time()
                answer = await self._rag._llm.generate(prompt)
                latency.generation_ms = (time.time() - gen_start) * 1000
                model_name = "llm-direct"
                intent_name = "general_qa"

        # Build response
        latency.total_ms = (time.time() - total_start) * 1000

        entities_dicts = [
            {"text": e.text, "type": e.type, "value": e.value, "confidence": e.confidence}
            for e in nlu_result.entities
        ]

        response = ChatResponse(
            answer=answer,
            session_id=session_id,
            intent=intent_name,
            entities=entities_dicts,
            sentiment=nlu_result.sentiment.value,
            sentiment_score=nlu_result.sentiment_score,
            citations=citations,
            retrieved_chunks=chunks,
            cache_hit=cache_hit,
            latency=latency,
            model_name=model_name,
            request_id=request_id,
        )

        # Save turn
        await self._sessions.add_turn(session_id, text, answer)

        # Cache response
        await self._cache.set(
            cache_key,
            response.model_dump(mode="json"),
            ttl=300,
        )

        # Emit analytics
        await self._bus.publish("chat.analytics", {
            "event_type": "chat.analytics",
            "session_id": session_id,
            "query": text[:200],
            "intent": intent_name,
            "latency_ms": latency.total_ms,
            "cache_hit": cache_hit,
            "rag_used": use_rag,
        })

        return response
