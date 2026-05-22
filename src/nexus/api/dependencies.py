"""
Dependencies — FastAPI dependency injection for application services.

This is the composition root: it builds and wires together all
infrastructure and application layer components.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any

import structlog
from fastapi import Request

from nexus.config import get_config

logger = structlog.get_logger(__name__)

# Singleton service instances (initialized during app lifespan)
_services: dict[str, Any] = {}


async def initialize_services() -> dict[str, Any]:
    """Build and wire all services. Called once at startup."""
    config = get_config()

    # ---- Cache ----
    if config.cache.backend == "redis":
        from nexus.infrastructure.cache.redis_cache import RedisCache
        cache = RedisCache(redis_url=config.cache.redis_url, default_ttl=config.cache.default_ttl)
    else:
        from nexus.infrastructure.cache.memory_cache import MemoryCache
        cache = MemoryCache(default_ttl=config.cache.default_ttl)

    # ---- Vector Store ----
    vector_store_type = getattr(config, 'vector_store_backend', 'qdrant')
    if vector_store_type == "faiss":
        from nexus.infrastructure.vectorstores.faiss_store import FAISSVectorStore
        vector_store = FAISSVectorStore(embedding_dim=768)
    else:
        from nexus.infrastructure.vectorstores.qdrant_store import QdrantVectorStore
        qdrant_url = getattr(config, 'qdrant_url', 'http://localhost:6333')
        vector_store = QdrantVectorStore(url=qdrant_url, embedding_dim=768)

    try:
        await vector_store.initialize()
    except Exception as e:
        logger.warning("vector_store_init_failed", error=str(e), fallback="continuing without vector store")

    # ---- Embedding ----
    from nexus.infrastructure.embeddings.sentence_transformer_embedder import SentenceTransformerEmbedder
    embedder = SentenceTransformerEmbedder(
        model_name=config.nlu.embedding_model,
        device=config.nlu.device,
    )

    # ---- LLM ----
    llm_provider_type = getattr(config, 'llm_provider', 'local')
    if llm_provider_type == "openai":
        from nexus.infrastructure.llm.openai_compatible_provider import OpenAICompatibleProvider
        llm = OpenAICompatibleProvider(
            base_url=getattr(config, 'openai_base_url', 'https://api.openai.com/v1'),
            api_key=getattr(config, 'openai_api_key', ''),
            model=getattr(config, 'openai_model', 'gpt-3.5-turbo'),
        )
    else:
        from nexus.infrastructure.llm.local_hf_provider import LocalHuggingFaceProvider
        llm = LocalHuggingFaceProvider(
            model_name=getattr(config, 'llm_model', 'gpt2'),
            device=config.nlu.device,
        )

    # ---- Message Bus ----
    kafka_enabled = getattr(config, 'kafka_enabled', False)
    if kafka_enabled:
        from nexus.infrastructure.messaging.kafka_producer import KafkaProducer
        bus = KafkaProducer(bootstrap_servers=getattr(config, 'kafka_bootstrap_servers', 'localhost:9092'))
        await bus.initialize()
    else:
        from nexus.infrastructure.messaging.kafka_producer import InMemoryMessageBus
        bus = InMemoryMessageBus()

    # ---- Repository ----
    from nexus.infrastructure.persistence.repositories import InMemoryDocumentRepository
    repository = InMemoryDocumentRepository()

    # ---- NLU ----
    from nexus.nlu.classifier import IntentClassifier
    from nexus.nlu.extractor import EntityExtractor
    from nexus.nlu.sentiment import SentimentAnalyzer
    from nexus.application.nlu_service import NLUService

    nlu_service = NLUService(
        classifier=IntentClassifier(config.nlu),
        extractor=EntityExtractor(config.nlu),
        analyzer=SentimentAnalyzer(config.nlu),
    )

    # Initialize models concurrently
    await asyncio.gather(
        nlu_service.initialize(),
        embedder.initialize(),
    )

    # Try LLM init (non-blocking failure)
    try:
        await llm.initialize()
    except Exception as e:
        logger.warning("llm_init_failed", error=str(e))

    # ---- Dialogue Manager ----
    from nexus.nlu.embeddings import EmbeddingEngine
    from nexus.dialogue.manager import DialogueManager

    dialogue_mgr = DialogueManager(config=config.dialogue)
    await dialogue_mgr.load()

    # ---- Application Services ----
    from nexus.application.rag_service import RAGService
    from nexus.application.session_service import SessionService
    from nexus.application.ingestion_service import IngestionService
    from nexus.application.chat_service import ChatService

    session_service = SessionService(cache=cache, session_ttl=config.dialogue.session_timeout_minutes * 60)
    rag_service = RAGService(vector_store=vector_store, embedder=embedder, llm=llm, cache=cache)
    ingestion_service = IngestionService(
        vector_store=vector_store, embedder=embedder, cache=cache,
        message_bus=bus, repository=repository,
    )
    chat_service = ChatService(
        nlu=nlu_service, rag=rag_service, sessions=session_service,
        dialogue_manager=dialogue_mgr, cache=cache, message_bus=bus,
    )

    services = {
        "cache": cache,
        "vector_store": vector_store,
        "embedder": embedder,
        "llm": llm,
        "bus": bus,
        "repository": repository,
        "nlu_service": nlu_service,
        "session_service": session_service,
        "rag_service": rag_service,
        "ingestion_service": ingestion_service,
        "chat_service": chat_service,
        "dialogue_manager": dialogue_mgr,
    }

    _services.update(services)
    return services


async def shutdown_services() -> None:
    """Gracefully shut down all services."""
    for name in ("cache", "vector_store", "bus"):
        svc = _services.get(name)
        if svc and hasattr(svc, "close"):
            try:
                await svc.close()
            except Exception as e:
                logger.warning(f"{name}_shutdown_error", error=str(e))


def get_chat_service(request: Request):
    return request.app.state.services["chat_service"]

def get_session_service(request: Request):
    return request.app.state.services["session_service"]

def get_ingestion_service(request: Request):
    return request.app.state.services["ingestion_service"]

def get_cache(request: Request):
    return request.app.state.services["cache"]

def get_repository(request: Request):
    return request.app.state.services["repository"]

def get_vector_store(request: Request):
    return request.app.state.services["vector_store"]

def get_message_bus(request: Request):
    return request.app.state.services["bus"]
