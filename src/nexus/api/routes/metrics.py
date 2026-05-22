"""Metrics route — /metrics (Prometheus exposition)"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response

from nexus.infrastructure.observability.metrics import get_metrics, get_content_type

router = APIRouter(tags=["Metrics"])


@router.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    return Response(content=get_metrics(), media_type=get_content_type())
