"""
FastAPI Application Factory
===========================

Creates and configures the FastAPI application with all middleware,
routes, and event handlers.
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
from nexus.core.engine import ConversationEngine
from nexus.api.routes import router
from nexus.api.websocket import websocket_router

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.
    
    Handles startup and shutdown events, including model loading.
    """
    logger.info("application_starting")
    
    # Initialize conversation engine
    config = get_config()
    engine = ConversationEngine(config)
    
    try:
        await engine.initialize()
        app.state.engine = engine
        logger.info("application_ready")
        yield
    finally:
        logger.info("application_shutting_down")
        # Cleanup
        await engine.cleanup_sessions()


def create_app(config: NexusConfig | None = None) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        config: Optional configuration override
    
    Returns:
        FastAPI: Configured application instance
    """
    config = config or get_config()
    
    app = FastAPI(
        title="Nexus Conversational AI",
        description=(
            "Enterprise-grade Conversational AI Engine with transformer-based NLU, "
            "multi-turn dialogue management, and real-time analytics."
        ),
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    
    # Store config
    app.state.config = config
    
    # Add middleware
    _add_middleware(app, config)
    
    # Add routes
    app.include_router(router, prefix="/api/v1")
    app.include_router(websocket_router, prefix="/ws")
    
    # Add exception handlers
    _add_exception_handlers(app)
    
    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check() -> dict:
        """Check application health status."""
        return {
            "status": "healthy",
            "version": "2.0.0",
            "engine_initialized": hasattr(app.state, "engine") and app.state.engine.is_initialized,
        }
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root() -> dict:
        """API root with basic information."""
        return {
            "name": "Nexus Conversational AI",
            "version": "2.0.0",
            "docs": "/docs",
            "health": "/health",
        }
    
    return app


def _add_middleware(app: FastAPI, config: NexusConfig) -> None:
    """Add middleware to the application."""
    
    # CORS
    if config.api.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.api.allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # GZip compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Request logging and timing
    @app.middleware("http")
    async def log_requests(request: Request, call_next: Callable) -> Response:
        """Log requests and add timing headers."""
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Add timing header
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        
        # Log request
        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        
        return response


def _add_exception_handlers(app: FastAPI) -> None:
    """Add custom exception handlers."""
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle uncaught exceptions."""
        logger.error(
            "unhandled_exception",
            path=request.url.path,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": "An unexpected error occurred. Please try again later.",
            },
        )


# Create default app instance
app = create_app()
