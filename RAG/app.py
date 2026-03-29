"""
app.py — FastAPI application factory
======================================
Single responsibility: assemble the application — register routers, configure
the lifespan context, and expose the ASGI `app` object.

No business logic lives here. This file has one reason to change: application
wiring (adding/removing routers, changing startup/shutdown behaviour).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from dependencies import get_cache_client
from ingestion import ingestion_service
from models import get_embeddings
from routers import health, ingest, query
from store import vector_store_manager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup:
      1. Load a persisted FAISS index from the Docker volume (survives restarts).
      2. Auto-ingest any documents pre-placed in the docs volume.
      3. Log semantic cache service availability (degraded mode is acceptable).
    Shutdown: nothing required — background threads are daemons.
    """
    vector_store_manager.load_persisted(get_embeddings())
    ingestion_service.auto_ingest()

    if not vector_store_manager.is_ready:
        logger.info("No documents loaded — POST /ingest to add documents")

    cache_ok = get_cache_client().is_available()
    logger.info(
        "Semantic cache service: %s",
        "available" if cache_ok else "unavailable (degraded mode — RAG still works)",
    )
    yield


app = FastAPI(
    title="RAG Workflow API",
    description=(
        "Retrieval-Augmented Generation over domain documents. "
        "Ollama + FAISS + LangChain. Semantic caching via semcache-api (separate service)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(query.router)
app.include_router(ingest.router)
app.include_router(health.router)
