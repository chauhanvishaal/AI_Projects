"""
api.py — FastAPI HTTP layer for the Semantic Cache pipeline
===========================================================
Exposes the SemanticCache as a lightweight REST API so that any downstream
agent workflow (any language / framework) can use semantic caching over HTTP.

Security:
  - All mutating / read endpoints require the X-API-Key header.
  - GET /health is intentionally unauthenticated (for Docker healthchecks).
  - Keys are compared with secrets.compare_digest to prevent timing attacks.

Start with:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

from semCache import SemanticCache, SemanticCacheConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

_cache: Optional[SemanticCache] = None
_config: Optional[SemanticCacheConfig] = None

API_KEY_ENV = os.environ.get("API_KEY", "")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")


# ---------------------------------------------------------------------------
# Lifespan — bootstrap cache on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cache, _config
    _config = SemanticCacheConfig()
    logger.info("Bootstrapping SemanticCache (FAISS rebuild from Redis)…")
    _cache = SemanticCache(_config)
    logger.info(
        "SemanticCache ready — %d vectors loaded into FAISS", _cache.faiss_vector_count
    )
    yield
    # Cleanup (nothing required — Pub/Sub thread is daemonized)


app = FastAPI(
    title="Semantic Cache API",
    description="Distributed semantic caching for AI agent workflows.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _require_api_key(api_key: str = Security(_api_key_header)) -> None:
    """Dependency that enforces X-API-Key authentication (timing-safe)."""
    if not API_KEY_ENV:
        # No key configured — open access (warn loudly)
        logger.warning("API_KEY is not set; running without authentication")
        return
    if not api_key or not secrets.compare_digest(api_key, API_KEY_ENV):
        raise HTTPException(status_code=403, detail="Invalid or missing API key")


def _get_cache() -> SemanticCache:
    if _cache is None:
        raise HTTPException(status_code=503, detail="Cache not initialised")
    return _cache


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The natural-language query")
    agent_id: str = Field(..., min_length=1, description="Logical agent identifier")
    ttl: Optional[int] = Field(None, gt=0, description="TTL in seconds (uses config default if omitted)")


class QueryResponse(BaseModel):
    response: str
    cache_hit: bool
    key: Optional[str] = None
    latency_ms: float


class InvalidateKeyResponse(BaseModel):
    deleted: bool


class InvalidateAgentResponse(BaseModel):
    deleted_count: int


class StoreRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The natural-language query to cache")
    response: str = Field(..., min_length=1, description="The pre-computed response to store")
    agent_id: str = Field(..., min_length=1, description="Logical agent identifier")
    ttl: Optional[int] = Field(None, gt=0, description="TTL in seconds (uses config default if omitted)")


class StoreResponse(BaseModel):
    key: str


class StatsResponse(BaseModel):
    agent_id: str
    hits: int
    misses: int
    total: int
    hit_rate: float


class HealthResponse(BaseModel):
    redis: str
    ollama: str
    faiss_vectors: int


# ---------------------------------------------------------------------------
# Helper — call Ollama for cache misses
# ---------------------------------------------------------------------------

def _call_ollama(query: str) -> str:
    """Send a generate request to the local Ollama server and return the text."""
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {"model": OLLAMA_MODEL, "prompt": query, "stream": False}
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "")
    except httpx.HTTPStatusError as exc:
        logger.error("Ollama HTTP error: %s", exc)
        raise HTTPException(status_code=502, detail=f"Ollama error: {exc.response.text}")
    except httpx.RequestError as exc:
        logger.error("Ollama connection error: %s", exc)
        raise HTTPException(status_code=502, detail="Could not reach Ollama")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
def query(
    body: QueryRequest,
    _: None = Depends(_require_api_key),
    cache: SemanticCache = Depends(_get_cache),
) -> QueryResponse:
    """
    Resolve a query through the semantic cache.

    - On a **cache hit** (cosine ≥ threshold): returns the cached response immediately.
    - On a **cache miss**: calls Ollama under a distributed lock, stores the result,
      and returns it.
    """
    start = time.perf_counter()

    cached_response = cache.get(body.query, body.agent_id)
    if cached_response is not None:
        latency_ms = (time.perf_counter() - start) * 1000
        return QueryResponse(
            response=cached_response,
            cache_hit=True,
            key=None,
            latency_ms=round(latency_ms, 2),
        )

    # Cache miss — use distributed lock to avoid stampede
    def compute() -> str:
        return _call_ollama(body.query)

    response = cache.cached_or_compute(
        query=body.query,
        agent_id=body.agent_id,
        compute_fn=compute,
        ttl=body.ttl,
    )

    latency_ms = (time.perf_counter() - start) * 1000
    return QueryResponse(
        response=response,
        cache_hit=False,
        key=None,
        latency_ms=round(latency_ms, 2),
    )


@app.delete("/invalidate/{key:path}", response_model=InvalidateKeyResponse)
def invalidate_key(
    key: str,
    _: None = Depends(_require_api_key),
    cache: SemanticCache = Depends(_get_cache),
) -> InvalidateKeyResponse:
    """Delete a specific cache entry by its Redis key."""
    deleted = cache.invalidate(key)
    return InvalidateKeyResponse(deleted=deleted)


@app.delete("/invalidate/agent/{agent_id}", response_model=InvalidateAgentResponse)
def invalidate_agent(
    agent_id: str,
    _: None = Depends(_require_api_key),
    cache: SemanticCache = Depends(_get_cache),
) -> InvalidateAgentResponse:
    """Delete all cached entries for a given agent."""
    count = cache.invalidate_by_agent(agent_id)
    return InvalidateAgentResponse(deleted_count=count)


@app.put("/cache", response_model=StoreResponse, status_code=201)
def store(
    body: StoreRequest,
    _: None = Depends(_require_api_key),
    cache: SemanticCache = Depends(_get_cache),
) -> StoreResponse:
    """
    Store a pre-computed (query, response) pair directly in the cache.

    Used by external services (e.g. the RAG workflow) that compute the answer
    themselves and want to populate the semantic cache for future callers.
    """
    key = cache.set(body.query, body.response, body.agent_id, body.ttl)
    return StoreResponse(key=key)


@app.get("/stats/{agent_id}", response_model=StatsResponse)
def stats(
    agent_id: str,
    _: None = Depends(_require_api_key),
    cache: SemanticCache = Depends(_get_cache),
) -> StatsResponse:
    """Return hit/miss statistics for a given agent."""
    s = cache.get_stats(agent_id)
    return StatsResponse(agent_id=agent_id, **s)


@app.get("/health", response_model=HealthResponse)
def health(cache: SemanticCache = Depends(_get_cache)) -> HealthResponse:
    """
    Health check — intentionally unauthenticated for Docker healthchecks.
    Probes Redis connectivity and Ollama reachability.
    """
    redis_status = "ok" if cache.redis_ping() else "fail"

    ollama_status = "fail"
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if resp.status_code == 200:
                ollama_status = "ok"
    except Exception:
        pass

    return HealthResponse(
        redis=redis_status,
        ollama=ollama_status,
        faiss_vectors=cache.faiss_vector_count,
    )
