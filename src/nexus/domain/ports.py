"""
Port interfaces (abstract base classes) for infrastructure adapters.

Application services depend on these abstractions, not concrete implementations.
This enables swapping backends (Qdrant ↔ FAISS, Redis ↔ in-memory, etc.)
without touching business logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator


class VectorStorePort(ABC):
    """Interface for vector storage backends (Qdrant, FAISS, etc.)."""

    @abstractmethod
    async def initialize(self) -> None:
        """Set up connection / load index."""

    @abstractmethod
    async def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        texts: list[str],
    ) -> None:
        """Insert or update vectors with metadata."""

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return top-k nearest neighbours with scores and metadata."""

    @abstractmethod
    async def delete(self, ids: list[str]) -> None:
        """Delete vectors by ID."""

    @abstractmethod
    async def delete_by_metadata(self, filters: dict[str, Any]) -> int:
        """Delete all vectors matching metadata filters. Returns count deleted."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the backend is reachable."""

    @abstractmethod
    async def close(self) -> None:
        """Release resources."""


class EmbeddingPort(ABC):
    """Interface for text embedding models."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the embedding dimension."""

    @abstractmethod
    async def initialize(self) -> None:
        """Load the model."""


class LLMProviderPort(ABC):
    """Interface for language-model text generation."""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a completion for the given prompt."""

    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Stream tokens for the given prompt."""
        yield ""  # pragma: no cover

    @abstractmethod
    async def initialize(self) -> None:
        """Load model / establish connection."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if provider is ready."""


class CachePort(ABC):
    """Interface for caching backends."""

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Get a value by key. Returns None on miss."""

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Set a value with optional TTL in seconds."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a key."""

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if cache is reachable."""

    @abstractmethod
    async def close(self) -> None:
        """Release connections."""


class MessageBusPort(ABC):
    """Interface for async message publishing."""

    @abstractmethod
    async def publish(self, topic: str, event: dict[str, Any]) -> None:
        """Publish an event to a topic."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if broker is reachable."""

    @abstractmethod
    async def close(self) -> None:
        """Release connections."""


class DocumentRepositoryPort(ABC):
    """Interface for document persistence."""

    @abstractmethod
    async def save(self, document: Any) -> None:
        """Persist a document record."""

    @abstractmethod
    async def get(self, document_id: str) -> Any | None:
        """Retrieve a document by ID."""

    @abstractmethod
    async def update_status(self, document_id: str, status: str, error: str | None = None) -> None:
        """Update document ingestion status."""

    @abstractmethod
    async def delete(self, document_id: str) -> None:
        """Delete a document record."""

    @abstractmethod
    async def list_all(self, limit: int = 100, offset: int = 0) -> list[Any]:
        """List documents with pagination."""
