# syntax=docker/dockerfile:1
# -------------------------------------------------------------------------
# Stage 1 — builder: compile native extensions (faiss-cpu, hiredis)
# -------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps needed to compile faiss-cpu and hiredis C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install into a prefix so we can copy cleanly to the runtime stage
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# -------------------------------------------------------------------------
# Stage 2 — runtime: minimal image with only what's needed at runtime
# -------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY semCache.py api.py example_usage.py ./

# Expose the API port
EXPOSE 8000

# Non-root user for security hardening
RUN useradd --no-create-home --shell /bin/false appuser \
    && chown -R appuser /app
USER appuser

# Uvicorn serves the FastAPI app
# Override DEMO_MODE env var to "langchain" or "http" to run example_usage.py instead
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
