# --- Stage 1: Get uv binary from official image ---
FROM ghcr.io/astral-sh/uv:0.6.6 AS uv

# --- Stage 2: Build dependencies and app ---
FROM python:3.14-slim AS builder

COPY --from=uv /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies first (cached unless pyproject.toml/uv.lock change)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source and sync the project itself
COPY . .
RUN uv sync --frozen --no-dev --no-editable

# --- Stage 3: Minimal runtime image (no build tools, no uv, no pip) ---
FROM python:3.14-slim AS runtime

# curl is needed for the healthcheck
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd -g 1000 app && useradd -u 1000 -g app -s /sbin/nologin user

WORKDIR /app

# Copy only the built app from builder, owned by non-root user
COPY --from=builder --chown=user:app /app /app

ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8082

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["curl", "-f", "http://localhost:8082/health"]

USER user

# Run uvicorn directly for proper signal handling (SIGTERM/SIGINT)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8082", "--timeout-graceful-shutdown", "5"]
