"""
API route modules.

Re-exports the legacy router and models for backward compatibility with existing
imports like `from nexus.api.routes import router`.
"""

from nexus.api.routes.legacy import (
    router,
    MessageRequest,
    MessageResponse,
    SessionResponse,
    SessionHistoryResponse,
    AnalyzeRequest,
    AnalyzeResponse,
)

__all__ = [
    "router",
    "MessageRequest",
    "MessageResponse",
    "SessionResponse",
    "SessionHistoryResponse",
    "AnalyzeRequest",
    "AnalyzeResponse",
]
