"""
workflow.py — backward-compatibility shim
==========================================
The application has been refactored into focused, single-responsibility modules:

  config.py       — Typed settings (env vars)
  schemas.py      — Pydantic request / response models
  models.py       — Ollama LLM + embeddings singletons
  store.py        — VectorStoreManager (FAISS persistence lifecycle)
  ingestion.py    — DocumentIngestionService (load, chunk, embed, index)
  rag_chain.py    — RAGChain (retrieve + generate)
  auth.py         — X-API-Key FastAPI dependency
  dependencies.py — Shared injectable dependencies
  routers/        — HTTP route handlers (query, ingest, health)
  app.py          — FastAPI app assembly + lifespan

This file re-exports `app` to maintain backward compatibility with any tooling
that still references `workflow:app`.  Prefer `uvicorn app:app` for new deployments.
"""

from app import app  # noqa: F401  -- re-exported for backward compatibility
