"""
routers/query.py — POST /query
================================
Single responsibility: handle question-answering requests by orchestrating
cache lookup → RAG chain execution → cache population.

No embedding logic, no FAISS access, no document loading.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException

from auth import require_api_key
from cache_client import SemanticCacheClient
from dependencies import get_cache_client
from rag_chain import rag_chain
from schemas import QueryRequest, QueryResponse

router = APIRouter(tags=["Query"])


@router.post("/query", response_model=QueryResponse)
def query(
    body:  QueryRequest,
    _:     None                = Depends(require_api_key),
    cache: SemanticCacheClient = Depends(get_cache_client),
) -> QueryResponse:
    """
    Answer a question via RAG with semantic cache integration.

    Flow:
      1. cache.lookup()  → HIT  → return instantly (~ms latency)
      2. Cache MISS      → rag_chain.run()  (FAISS retrieval + Ollama LLM)
      3. cache.store()   → populate cache for future similar questions
    """
    start = time.perf_counter()

    # Step 1 — cache check (delegated entirely to the cache client)
    if body.use_cache:
        cached_answer = cache.lookup(body.question)
        if cached_answer is not None:
            return QueryResponse(
                answer=cached_answer,
                sources=[],
                chunks_retrieved=0,
                cache_hit=True,
                latency_ms=round((time.perf_counter() - start) * 1000, 2),
            )

    # Step 2 — RAG chain (this router's only real concern)
    try:
        result = rag_chain.run(body.question, body.top_k)
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"{exc} — POST /ingest to add documents first",
        )

    # Step 3 — populate cache for future callers (fire-and-forget; never blocks)
    if body.use_cache:
        cache.store(body.question, result["answer"])

    return QueryResponse(
        **result,
        cache_hit=False,
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )
