"""
Domain-level exceptions.

These are raised by application services and caught by the API layer
to produce appropriate HTTP error responses.
"""

from __future__ import annotations


class NexusError(Exception):
    """Base exception for all Nexus domain errors."""

    def __init__(self, message: str = "An unexpected error occurred", code: str = "NEXUS_ERROR"):
        self.message = message
        self.code = code
        super().__init__(message)


class DocumentNotFoundError(NexusError):
    def __init__(self, document_id: str):
        super().__init__(
            message=f"Document not found: {document_id}",
            code="DOCUMENT_NOT_FOUND",
        )


class SessionNotFoundError(NexusError):
    def __init__(self, session_id: str):
        super().__init__(
            message=f"Session not found: {session_id}",
            code="SESSION_NOT_FOUND",
        )


class IngestionError(NexusError):
    def __init__(self, message: str = "Document ingestion failed"):
        super().__init__(message=message, code="INGESTION_ERROR")


class VectorStoreError(NexusError):
    def __init__(self, message: str = "Vector store operation failed"):
        super().__init__(message=message, code="VECTOR_STORE_ERROR")


class LLMProviderError(NexusError):
    def __init__(self, message: str = "LLM generation failed"):
        super().__init__(message=message, code="LLM_PROVIDER_ERROR")


class RateLimitError(NexusError):
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message=message, code="RATE_LIMIT_EXCEEDED")


class CacheError(NexusError):
    def __init__(self, message: str = "Cache operation failed"):
        super().__init__(message=message, code="CACHE_ERROR")


class ServiceUnavailableError(NexusError):
    def __init__(self, service: str = "unknown"):
        super().__init__(
            message=f"Service unavailable: {service}",
            code="SERVICE_UNAVAILABLE",
        )


class PromptInjectionError(NexusError):
    def __init__(self, message: str = "Potential prompt injection detected"):
        super().__init__(message=message, code="PROMPT_INJECTION")
