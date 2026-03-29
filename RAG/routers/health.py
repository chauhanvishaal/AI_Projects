"""
routers/health.py — GET /health
=================================
Single responsibility: report the liveness of all external dependencies.

Intentionally unauthenticated — safe for Docker HEALTHCHECK and orchestrators.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends

from cache_client import SemanticCacheClient
from config import settings
from dependencies import get_cache_client
from schemas import HealthResponse
from store import vector_store_manager

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health(cache: SemanticCacheClient = Depends(get_cache_client)) -> HealthResponse:
    """
    Health check — unauthenticated, safe for Docker HEALTHCHECK and orchestrators.
    Probes Ollama reachability, semantic cache availability, and FAISS vector count.
    """
    ollama_status = "fail"
    try:
        with httpx.Client(timeout=5.0) as client:
            if client.get(f"{settings.ollama_base_url}/api/tags").status_code == 200:
                ollama_status = "ok"
    except Exception:
        pass

    return HealthResponse(
        ollama=ollama_status,
        semantic_cache="ok" if cache.is_available() else "unavailable",
        faiss_vectors=vector_store_manager.vector_count,
        docs_path=settings.docs_path,
    )
