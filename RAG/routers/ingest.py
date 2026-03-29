"""
routers/ingest.py — POST /ingest, DELETE /index
=================================================
Single responsibility: handle document upload and vector index management.

No RAG chain logic, no cache logic.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile

from auth import require_api_key
from config import settings
from ingestion import ingestion_service
from schemas import IngestResponse
from store import vector_store_manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Ingestion"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    files: list[UploadFile] = File(...),
    _:     None             = Depends(require_api_key),
) -> IngestResponse:
    """
    Upload one or more documents (PDF, TXT, MD) to be chunked, embedded,
    and added to the persisted FAISS vector store.
    """
    start = time.perf_counter()
    docs_dir = Path(settings.docs_path)
    docs_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for upload in files:
        content = await upload.read()
        dest = docs_dir / upload.filename
        dest.write_bytes(content)
        saved.append(dest)
        logger.info("Received upload: %s (%d bytes)", upload.filename, len(content))

    chunks_added = ingestion_service.ingest(saved)

    return IngestResponse(
        chunks_added=chunks_added,
        total_vectors=vector_store_manager.vector_count,
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )


@router.delete("/index", dependencies=[Depends(require_api_key)])
def clear_index() -> dict:
    """Wipe the FAISS vector store and delete the persisted index from disk."""
    vector_store_manager.clear()
    return {"cleared": True}
