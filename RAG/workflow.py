"""
workflow.py — Containerised RAG pipeline
=========================================
Single responsibility: ingest documents and answer questions via RAG.

Cache integration is fully delegated to SemanticCacheClient (cache_client.py).
This file contains zero cache logic — it has one reason to change: the RAG pipeline.

Stack (mirrors Semantic Cache project — fully free, self-hostable):
  - Ollama          : LLM (llama3) + embeddings (nomic-embed-text)
  - FAISS           : local vector store (faiss-cpu, disk-persisted)
  - LangChain       : loaders, splitter, retrieval chain
  - FastAPI         : REST API  (port 8001 on host)
  - semcache-api    : semantic cache service (separate container, port 8000)

Query flow:
  1. SemanticCacheClient.lookup()  → cache hit → return immediately
  2. Cache miss → FAISS retrieval + Ollama LLM
  3. SemanticCacheClient.store()   → populate cache for future callers
"""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from fastapi import Depends, FastAPI, File, HTTPException, Security, UploadFile
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaEmbeddings, OllamaLLM

from cache_client import SemanticCacheClient

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Config (env vars — same pattern as semCache)
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL  = os.environ.get("OLLAMA_BASE_URL",     "http://localhost:11434")
OLLAMA_MODEL     = os.environ.get("OLLAMA_MODEL",        "llama3")
EMBEDDING_MODEL  = os.environ.get("EMBEDDING_MODEL",     "nomic-embed-text")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH",    "/data/faiss_index")
DOCS_PATH        = os.environ.get("DOCS_PATH",           "/data/documents")
CHUNK_SIZE       = int(os.environ.get("CHUNK_SIZE",      "512"))
CHUNK_OVERLAP    = int(os.environ.get("CHUNK_OVERLAP",   "64"))
RETRIEVAL_TOP_K  = int(os.environ.get("RETRIEVAL_TOP_K", "4"))
API_KEY_ENV      = os.environ.get("API_KEY", "")

# ---------------------------------------------------------------------------
# Shared state (lazy-initialised singletons)
# ---------------------------------------------------------------------------

_vector_store: Optional[LangchainFAISS] = None
_embeddings:   Optional[OllamaEmbeddings] = None
_llm:          Optional[OllamaLLM] = None
_cache_client: Optional[SemanticCacheClient] = None


def _get_embeddings() -> OllamaEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    return _embeddings


def _get_llm() -> OllamaLLM:
    global _llm
    if _llm is None:
        _llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
    return _llm


def _get_cache_client() -> SemanticCacheClient:
    global _cache_client
    if _cache_client is None:
        _cache_client = SemanticCacheClient()
    return _cache_client


# ---------------------------------------------------------------------------
# FAISS persistence  (RAG concern: store document vectors between restarts)
# ---------------------------------------------------------------------------

def _load_persisted_store() -> Optional[LangchainFAISS]:
    """Load a previously saved FAISS index from the mounted volume."""
    index_dir = Path(FAISS_INDEX_PATH)
    if index_dir.exists() and any(index_dir.iterdir()):
        try:
            store = LangchainFAISS.load_local(
                str(index_dir),
                _get_embeddings(),
                allow_dangerous_deserialization=True,
            )
            logger.info("FAISS index loaded — %d vectors", store.index.ntotal)
            return store
        except Exception as exc:
            logger.warning("Could not load FAISS index: %s — starting fresh", exc)
    return None


def _save_store(store: LangchainFAISS) -> None:
    Path(FAISS_INDEX_PATH).mkdir(parents=True, exist_ok=True)
    store.save_local(FAISS_INDEX_PATH)
    logger.info("FAISS index saved (%d vectors)", store.index.ntotal)


def _require_store() -> LangchainFAISS:
    if _vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="No documents ingested yet — POST /ingest to add documents",
        )
    return _vector_store


# ---------------------------------------------------------------------------
# Document ingestion  (RAG concern: chunk + embed domain documents)
# ---------------------------------------------------------------------------

def _loader_for(path: Path):
    """Choose the right LangChain loader by file extension."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path))
    if ext in (".md", ".markdown"):
        return UnstructuredMarkdownLoader(str(path))
    return TextLoader(str(path), encoding="utf-8")


def _ingest_documents(file_paths: list[Path]) -> int:
    """Chunk, embed, and add documents to the local FAISS index."""
    global _vector_store

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    all_chunks: list[Document] = []
    for path in file_paths:
        try:
            docs = _loader_for(path).load()
            chunks = splitter.split_documents(docs)
            for chunk in chunks:
                chunk.metadata.setdefault("source", path.name)
            all_chunks.extend(chunks)
            logger.info("Loaded %s — %d chunks", path.name, len(chunks))
        except Exception as exc:
            logger.warning("Failed to load %s: %s", path, exc)

    if not all_chunks:
        return 0

    if _vector_store is None:
        _vector_store = LangchainFAISS.from_documents(all_chunks, _get_embeddings())
    else:
        _vector_store.add_documents(all_chunks)

    _save_store(_vector_store)
    return len(all_chunks)


def _auto_ingest_docs_path() -> None:
    """On startup, ingest any documents pre-loaded into DOCS_PATH."""
    docs_dir = Path(DOCS_PATH)
    if not docs_dir.exists():
        return
    supported = {".pdf", ".txt", ".md", ".markdown", ".text"}
    files = [p for p in docs_dir.rglob("*") if p.suffix.lower() in supported and p.is_file()]
    if not files:
        return
    logger.info("Auto-ingesting %d file(s) from %s", len(files), DOCS_PATH)
    _ingest_documents(files)


# ---------------------------------------------------------------------------
# RAG chain  (RAG concern: retrieve + generate — no cache logic here)
# ---------------------------------------------------------------------------

_RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not contained in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
)


def _format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in docs
    )


def _run_rag(question: str, top_k: int) -> dict:
    """Retrieve relevant chunks and generate an answer. Pure RAG — no cache logic."""
    store = _require_store()
    retriever = store.as_retriever(search_kwargs={"k": top_k})

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | _RAG_PROMPT
        | _get_llm()
        | StrOutputParser()
    )

    retrieved_docs = retriever.invoke(question)
    answer = chain.invoke(question)
    sources = sorted({d.metadata.get("source", "unknown") for d in retrieved_docs})
    return {"answer": answer, "sources": sources, "chunks_retrieved": len(retrieved_docs)}


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _vector_store
    # Load persisted FAISS index (survives container restarts via Docker volume)
    _vector_store = _load_persisted_store()
    # Pick up any documents pre-placed in the documents volume
    _auto_ingest_docs_path()
    if _vector_store is None:
        logger.info("No documents loaded — POST /ingest to add documents")
    # Report cache service status (degraded mode is fine if unavailable)
    cache_ok = _get_cache_client().is_available()
    logger.info("Semantic cache service: %s", "available" if cache_ok else "unavailable (degraded mode — RAG still works)")
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

# ---------------------------------------------------------------------------
# Authentication (same X-API-Key + secrets.compare_digest pattern as semCache)
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _require_api_key(api_key: str = Security(_api_key_header)) -> None:
    if not API_KEY_ENV:
        logger.warning("API_KEY is not set; running without authentication")
        return
    if not api_key or not secrets.compare_digest(api_key, API_KEY_ENV):
        raise HTTPException(status_code=403, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to answer from ingested documents")
    top_k: int = Field(default=RETRIEVAL_TOP_K, ge=1, le=20, description="Number of chunks to retrieve")
    use_cache: bool = Field(default=True, description="Set False to bypass the semantic cache and force a fresh RAG call")


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_retrieved: int
    cache_hit: bool
    latency_ms: float


class IngestResponse(BaseModel):
    chunks_added: int
    total_vectors: int
    latency_ms: float


class HealthResponse(BaseModel):
    ollama: str
    semantic_cache: str
    faiss_vectors: int
    docs_path: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, _: None = Depends(_require_api_key)) -> QueryResponse:
    """
    Answer a question via RAG with semantic cache integration.

    Flow (SRP — each step is a separate concern):
      1. cache_client.lookup()  → cache HIT  → return immediately (ms latency)
      2. Cache MISS             → _run_rag() → FAISS retrieval + Ollama LLM
      3. cache_client.store()   → populate cache for future identical/similar questions
    """
    start = time.perf_counter()
    cache = _get_cache_client()

    # Step 1 — delegate cache check entirely to the cache client
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

    # Step 2 — cache miss: run the RAG chain (this file's only real job)
    result = _run_rag(body.question, body.top_k)

    # Step 3 — populate cache for future callers (fire-and-forget; never blocks)
    if body.use_cache:
        cache.store(body.question, result["answer"])

    return QueryResponse(
        **result,
        cache_hit=False,
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    files: list[UploadFile] = File(...),
    _: None = Depends(_require_api_key),
) -> IngestResponse:
    """
    Upload one or more documents (PDF, TXT, MD) to be chunked, embedded,
    and added to the persisted FAISS vector store.
    """
    start = time.perf_counter()
    docs_dir = Path(DOCS_PATH)
    docs_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for upload in files:
        content = await upload.read()
        dest = docs_dir / upload.filename
        dest.write_bytes(content)
        saved.append(dest)
        logger.info("Received upload: %s (%d bytes)", upload.filename, len(content))

    chunks_added = _ingest_documents(saved)
    total_vectors = _vector_store.index.ntotal if _vector_store else 0

    return IngestResponse(
        chunks_added=chunks_added,
        total_vectors=total_vectors,
        latency_ms=round((time.perf_counter() - start) * 1000, 2),
    )


@app.delete("/index", dependencies=[Depends(_require_api_key)])
def clear_index() -> dict:
    """Wipe the FAISS vector store and delete the persisted index from disk."""
    global _vector_store
    _vector_store = None
    index_dir = Path(FAISS_INDEX_PATH)
    if index_dir.exists():
        shutil.rmtree(index_dir)
    logger.info("FAISS index cleared")
    return {"cleared": True}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check — unauthenticated, safe for Docker healthchecks and orchestrators."""
    ollama_status = "fail"
    try:
        with httpx.Client(timeout=5.0) as client:
            if client.get(f"{OLLAMA_BASE_URL}/api/tags").status_code == 200:
                ollama_status = "ok"
    except Exception:
        pass

    return HealthResponse(
        ollama=ollama_status,
        semantic_cache="ok" if _get_cache_client().is_available() else "unavailable",
        faiss_vectors=_vector_store.index.ntotal if _vector_store else 0,
        docs_path=DOCS_PATH,
    )
