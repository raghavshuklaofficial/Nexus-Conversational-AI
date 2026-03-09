# Multi-stage build for Nexus AI

# Stage 1: build wheels
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir build && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels .

# Stage 2: slim production image
FROM python:3.11-slim as production

WORKDIR /app

# non-root user
RUN groupadd --gid 1000 nexus && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home nexus

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy wheels and install
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copy application code
COPY --chown=nexus:nexus src/ ./src/
COPY --chown=nexus:nexus frontend/ ./frontend/

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    NEXUS_ENVIRONMENT=production \
    NEXUS_HOST=0.0.0.0 \
    NEXUS_PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to non-root user
USER nexus

# Run application
CMD ["python", "-m", "nexus.cli", "serve", "--host", "0.0.0.0", "--port", "8000"]

# Stage 3: Development
FROM production as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    ruff \
    mypy

# Copy test files
COPY --chown=nexus:nexus tests/ ./tests/

USER nexus

CMD ["python", "-m", "pytest", "tests/", "-v"]
