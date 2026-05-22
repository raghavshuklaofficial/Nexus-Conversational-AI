"""
FastAPI app — creates the application with middleware, modular routes,
and manages full service lifecycle (startup/shutdown).
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from nexus.config import NexusConfig, get_config
from nexus.infrastructure.observability.tracing import (
    bind_request_context,
    clear_request_context,
    generate_request_id,
)
from nexus.infrastructure.observability.metrics import (
    APP_INFO, REQUEST_COUNT, REQUEST_LATENCY,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: initialize all services. Shutdown: graceful cleanup."""
    logger.info("application_starting")

    from nexus.api.dependencies import initialize_services, shutdown_services

    try:
        services = await initialize_services()
        app.state.services = services

        # Expose chat_service on app.state for WebSocket handler
        app.state.chat_service = services.get("chat_service")

        # Backward-compat engine reference for legacy routes
        # Legacy routes use request.app.state.engine via get_engine()
        from nexus.core.engine import ConversationEngine
        engine = ConversationEngine()
        await engine.initialize()
        app.state.engine = engine

        APP_INFO.info({"version": "3.0.0", "environment": get_config().environment.value})

        logger.info("application_ready")
        yield
    finally:
        logger.info("application_shutting_down")
        await shutdown_services()


def create_app(config: NexusConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    config = config or get_config()

    app = FastAPI(
        title="Nexus Conversational AI",
        description=(
            "Production-grade conversational AI platform with RAG, "
            "async ingestion, and multi-model NLU."
        ),
        version="3.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.state.config = config

    # ---- Middleware ----
    _add_middleware(app, config)

    # ---- Routes ----
    from nexus.api.routes.chat import router as chat_router
    from nexus.api.routes.documents import router as docs_router
    from nexus.api.routes.sessions import router as sessions_router
    from nexus.api.routes.health import router as health_router
    from nexus.api.routes.metrics import router as metrics_router

    # Keep original routes for backward compatibility
    from nexus.api.routes import router as legacy_router

    app.include_router(chat_router, prefix="/api/v1")
    app.include_router(docs_router, prefix="/api/v1")
    app.include_router(sessions_router, prefix="/api/v1")
    app.include_router(health_router)
    app.include_router(metrics_router)

    # Legacy routes
    app.include_router(legacy_router, prefix="/api/v1")

    # WebSocket
    from nexus.api.websocket import websocket_router
    app.include_router(websocket_router, prefix="/ws")

    # Exception handlers
    _add_exception_handlers(app)

    # Root endpoints
    @app.get("/health", tags=["Health"])
    async def health_check() -> dict:
        return {"status": "healthy", "version": "3.0.0"}

    from pathlib import Path
    frontend_dir = Path(__file__).resolve().parent.parent.parent.parent / "frontend"
    if frontend_dir.exists():
        from fastapi.responses import HTMLResponse
        from fastapi.staticfiles import StaticFiles

        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

        @app.get("/", tags=["Root"], response_class=HTMLResponse)
        async def root():
            return (frontend_dir / "index.html").read_text()
    else:
        @app.get("/", tags=["Root"])
        async def root() -> dict:
            return {
                "name": "Nexus Conversational AI",
                "version": "3.0.0",
                "docs": "/docs",
                "health": "/health",
            }

    return app


def _add_middleware(app: FastAPI, config: NexusConfig) -> None:
    """Add middleware stack."""
    if config.api.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.api.allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.add_middleware(GZipMiddleware, minimum_size=1000)

    @app.middleware("http")
    async def request_middleware(request: Request, call_next: Callable) -> Response:
        """Attach request ID, log, time, and record metrics."""
        request_id = generate_request_id()
        bind_request_context(request_id)

        start = time.time()
        response = await call_next(request)
        duration = time.time() - start

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration * 1000:.2f}ms"

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=str(response.status_code),
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path,
            status=str(response.status_code),
        ).observe(duration)

        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration * 1000, 2),
        )

        clear_request_context()
        return response


def _add_exception_handlers(app: FastAPI) -> None:
    """Add safe error responses."""
    from nexus.domain.errors import NexusError, RateLimitError

    @app.exception_handler(RateLimitError)
    async def rate_limit_handler(request: Request, exc: RateLimitError) -> JSONResponse:
        return JSONResponse(status_code=429, content={"error": exc.code, "message": exc.message})

    @app.exception_handler(NexusError)
    async def nexus_error_handler(request: Request, exc: NexusError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": exc.code, "message": exc.message})

    @app.exception_handler(Exception)
    async def global_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_exception", path=request.url.path, error=str(exc))
        return JSONResponse(
            status_code=500,
            content={"error": "internal_server_error", "message": "An unexpected error occurred."},
        )


# Create default app instance
app = create_app()
