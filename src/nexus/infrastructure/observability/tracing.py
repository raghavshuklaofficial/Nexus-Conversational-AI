"""
Request tracing — attaches request_id to structured logs via contextvars.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar

import structlog

request_id_var: ContextVar[str] = ContextVar("request_id", default="")
session_id_var: ContextVar[str] = ContextVar("session_id", default="")


def generate_request_id() -> str:
    return str(uuid.uuid4())[:8]


def bind_request_context(request_id: str, session_id: str = "") -> None:
    request_id_var.set(request_id)
    session_id_var.set(session_id)
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        session_id=session_id,
    )


def clear_request_context() -> None:
    structlog.contextvars.unbind_contextvars("request_id", "session_id")
