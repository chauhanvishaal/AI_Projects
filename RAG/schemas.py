"""
schemas.py — API request / response models
==========================================
Single responsibility: define the shape of data that crosses the HTTP boundary.
No business logic, no external I/O.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from config import settings


class QueryRequest(BaseModel):
    question:  str  = Field(..., min_length=1, description="Question to answer from ingested documents")
    top_k:     int  = Field(default=settings.retrieval_top_k, ge=1, le=20, description="Number of chunks to retrieve")
    use_cache: bool = Field(default=True, description="Set False to bypass the semantic cache and force a fresh RAG call")


class QueryResponse(BaseModel):
    answer:           str
    sources:          list[str]
    chunks_retrieved: int
    cache_hit:        bool
    latency_ms:       float


class IngestResponse(BaseModel):
    chunks_added:  int
    total_vectors: int
    latency_ms:    float


class HealthResponse(BaseModel):
    ollama:         str
    semantic_cache: str
    faiss_vectors:  int
    docs_path:      str
