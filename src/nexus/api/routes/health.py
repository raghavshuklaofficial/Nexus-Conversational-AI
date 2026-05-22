"""Health check routes — /health/live, /health/ready"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Request

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/live")
async def liveness():
    """Liveness probe — always returns OK if the process is running."""
    return {"status": "alive"}


@router.get("/ready")
async def readiness(request: Request):
    """Readiness probe — checks all dependencies."""
    services = getattr(request.app.state, "services", {})
    checks = {}

    # Cache
    cache = services.get("cache")
    if cache:
        checks["cache"] = await cache.health_check()

    # Vector store
    vs = services.get("vector_store")
    if vs:
        try:
            checks["vector_store"] = await vs.health_check()
        except Exception:
            checks["vector_store"] = False

    # Message bus
    bus = services.get("bus")
    if bus:
        checks["message_bus"] = await bus.health_check()

    # NLU
    nlu = services.get("nlu_service")
    if nlu:
        checks["nlu"] = nlu.is_ready

    # LLM
    llm = services.get("llm")
    if llm:
        checks["llm"] = await llm.health_check()

    all_ready = all(checks.values()) if checks else True
    status_code = 200 if all_ready else 503

    return {
        "status": "ready" if all_ready else "degraded",
        "checks": checks,
    }
