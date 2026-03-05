# Multi-stage Docker build for EvalKit
# =======================================

# ── Stage 1: Build dependencies ───────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build tools
RUN pip install --upgrade pip hatchling build

# Copy project files for dependency resolution
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Build the wheel
RUN python -m build --wheel --outdir /dist

# Install dependencies into a virtual environment for the final stage
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install /dist/*.whl

# ── Stage 2: Runtime image ────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Security: create a non-root user
RUN groupadd --gid 1001 evalkit && \
    useradd --uid 1001 --gid 1001 --no-create-home evalkit

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy dataset files and configs
COPY datasets/ ./datasets/
COPY configs/ ./configs/

# Create storage directories
RUN mkdir -p /app/evalkit_storage /app/evalkit_output && \
    chown -R evalkit:evalkit /app

# Switch to non-root user
USER evalkit

# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

# Environment variable defaults
ENV EVALKIT_STORAGE_PATH="/app/evalkit_storage"
ENV EVALKIT_OUTPUT_DIR="/app/evalkit_output"
ENV UVICORN_HOST="0.0.0.0"
ENV UVICORN_PORT="8000"
ENV UVICORN_WORKERS="1"

# Expose the API port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')" || exit 1

# Default command: run the API server
CMD ["evalkit", "serve", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--storage", "/app/evalkit_storage", \
     "--storage-backend", "sqlite"]
