# ─── Stage 1: builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies only in this stage
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt

# ─── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy only installed packages (no compiler toolchain in production image)
COPY --from=builder /install /usr/local

# Application source (SRP: only the two files that belong to this service)
COPY workflow.py cache_client.py ./

# Non-root user for security
RUN adduser --disabled-password --no-create-home appuser \
    && mkdir -p /data/faiss_index /data/documents \
    && chown -R appuser:appuser /data /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "workflow:app", "--host", "0.0.0.0", "--port", "8000"]
