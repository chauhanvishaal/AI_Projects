"""
example_usage.py — RAG Workflow API usage examples
====================================================

Demonstrates three usage patterns for the RAG Workflow API:

  1. INGEST   — upload documents (PDF, TXT, MD) to build the FAISS index
  2. QUERY    — ask a question; shows cache miss → RAG then cache hit → instant return
  3. HEALTH   — check service health (Ollama, FAISS, semcache-api)

Prerequisites:
  Option A — Docker Compose (recommended):
    cd d:\\dev\\py\\RAG
    cp .env.example .env   # fill in API keys
    docker compose up -d

  Option B — Local (no Docker):
    1. Redis:   docker run -d -p 6379:6379 redis:7-alpine
    2. Ollama:  ollama serve && ollama pull llama3 && ollama pull nomic-embed-text
    3. semcache-api:  cd ../SemanticCaching && uvicorn api:app --port 8000
    4. rag-api:       uvicorn workflow:app --port 8001

Environment variables (defaults shown):
  RAG_BASE_URL  = http://localhost:8001
  RAG_API_KEY   = (empty — no auth in dev)
  SAMPLE_DOCS   = ./sample_docs/  (directory of demo files to ingest)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Config  (read from env so this file works inside or outside Docker)
# ---------------------------------------------------------------------------

RAG_BASE_URL = os.environ.get("RAG_BASE_URL", "http://localhost:8001")
RAG_API_KEY  = os.environ.get("RAG_API_KEY", "")
SAMPLE_DOCS  = os.environ.get("SAMPLE_DOCS", str(Path(__file__).parent / "sample_docs"))

HEADERS = {
    "X-API-Key": RAG_API_KEY,
    "Accept": "application/json",
}

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def _fmt_ms(ms: float) -> str:
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.2f} s"


# ---------------------------------------------------------------------------
# 1. Health check
# ---------------------------------------------------------------------------

def demo_health(client: httpx.Client) -> bool:
    _print_section("1. Health check — GET /health")

    resp = client.get(f"{RAG_BASE_URL}/health")
    resp.raise_for_status()
    health = resp.json()

    print(f"  Ollama:         {health['ollama']}")
    print(f"  Semantic cache: {health['semantic_cache']}")
    print(f"  FAISS vectors:  {health['faiss_vectors']}")
    print(f"  Docs path:      {health['docs_path']}")

    ok = health["ollama"] == "ok"
    if not ok:
        print("\n  [!] Ollama is not reachable — start it before running queries.")
    return ok


# ---------------------------------------------------------------------------
# 2. Document ingestion
# ---------------------------------------------------------------------------

def demo_ingest(client: httpx.Client) -> int:
    """
    Uploads all supported files from SAMPLE_DOCS to POST /ingest.

    Creates a small sample text file if the directory doesn't exist so the
    demo runs end-to-end without requiring real documents.
    """
    _print_section("2. Document ingestion — POST /ingest")

    docs_dir = Path(SAMPLE_DOCS)
    if not docs_dir.exists() or not any(docs_dir.iterdir()):
        print(f"  No documents found in {docs_dir} — creating a sample file.")
        docs_dir.mkdir(parents=True, exist_ok=True)
        sample = docs_dir / "sample.txt"
        sample.write_text(
            "Retrieval-Augmented Generation (RAG) combines dense vector retrieval "
            "with a large language model. Given a user query, the retriever fetches "
            "the most semantically similar document chunks from a FAISS index. "
            "Those chunks are inserted into a prompt template alongside the question "
            "and the LLM generates a grounded answer.  This significantly reduces "
            "hallucinations compared to a vanilla LLM call.",
            encoding="utf-8",
        )

    supported = {".pdf", ".txt", ".md", ".markdown", ".text"}
    files = [p for p in docs_dir.rglob("*") if p.suffix.lower() in supported and p.is_file()]

    if not files:
        print(f"  No supported files found in {docs_dir}")
        return 0

    print(f"  Uploading {len(files)} file(s):")
    for f in files:
        print(f"    • {f.name}")

    multipart = [
        ("files", (p.name, p.read_bytes(), _mime_type(p)))
        for p in files
    ]

    t0 = time.perf_counter()
    resp = client.post(
        f"{RAG_BASE_URL}/ingest",
        headers={k: v for k, v in HEADERS.items() if k != "Accept"},
        files=multipart,
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"\n  Chunks added:   {data['chunks_added']}")
    print(f"  Total vectors:  {data['total_vectors']}")
    print(f"  Latency:        {_fmt_ms(elapsed)}")

    return data["total_vectors"]


def _mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".pdf":      "application/pdf",
        ".md":       "text/markdown",
        ".markdown": "text/markdown",
    }.get(ext, "text/plain")


# ---------------------------------------------------------------------------
# 3. Query (cache miss → RAG, then semantically similar → cache hit)
# ---------------------------------------------------------------------------

def demo_query(client: httpx.Client) -> None:
    _print_section("3. Querying — POST /query")

    questions = [
        ("What is Retrieval-Augmented Generation?",          True),
        # Semantically similar — should hit the semantic cache on the second call
        ("Can you explain what RAG means in AI?",            True),
        # Bypasses cache entirely — always runs the full RAG chain
        ("Summarise how RAG reduces hallucinations.",         False),
    ]

    for question, use_cache in questions:
        label = "cache-enabled" if use_cache else "cache bypassed"
        print(f"\n  Question ({label}):\n  {question!r}")

        t0 = time.perf_counter()
        resp = client.post(
            f"{RAG_BASE_URL}/query",
            headers=HEADERS,
            json={"question": question, "use_cache": use_cache, "top_k": 4},
            timeout=120.0,
        )

        if resp.status_code == 503:
            print("  [!] No documents ingested yet — run demo_ingest() first.")
            continue

        resp.raise_for_status()
        data = resp.json()
        elapsed = (time.perf_counter() - t0) * 1000

        hit_label = "HIT  (returned from semantic cache)" if data["cache_hit"] else "MISS (RAG chain executed)"
        print(f"  Cache: {hit_label}")
        print(f"  Chunks retrieved: {data['chunks_retrieved']}")
        if data["sources"]:
            print(f"  Sources: {', '.join(data['sources'])}")
        print(f"  Latency: {_fmt_ms(elapsed)}")
        answer_preview = data["answer"].replace("\n", " ")[:200]
        print(f"  Answer: {answer_preview}{'...' if len(data['answer']) > 200 else ''}")


# ---------------------------------------------------------------------------
# 4. Cache bypass comparison  (shows latency difference clearly)
# ---------------------------------------------------------------------------

def demo_latency_comparison(client: httpx.Client) -> None:
    _print_section("4. Latency comparison — cache vs RAG")

    question = "How does FAISS enable fast similarity search?"

    timings: dict[str, float] = {}

    for label, use_cache in [("RAG (first call, cache miss)", True),
                              ("Cache hit (same question)",    True),
                              ("RAG forced (cache bypassed)",  False)]:
        t0 = time.perf_counter()
        resp = client.post(
            f"{RAG_BASE_URL}/query",
            headers=HEADERS,
            json={"question": question, "use_cache": use_cache},
            timeout=120.0,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        if resp.status_code == 503:
            print(f"  {label}: skipped (no index)")
            continue
        resp.raise_for_status()
        data = resp.json()
        timings[label] = elapsed
        hit = "HIT" if data["cache_hit"] else "MISS"
        print(f"\n  {label}")
        print(f"    Cache: {hit} | API latency: {_fmt_ms(elapsed)}")

    if len(timings) >= 2:
        vals = list(timings.values())
        speedup = vals[0] / vals[1] if vals[1] > 0 else 0
        print(f"\n  Speed-up (miss → hit): {speedup:.1f}×")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    demo_mode = os.environ.get("DEMO_MODE", "all").lower()

    print(f"\nRAG Workflow API — example usage")
    print(f"  Target:    {RAG_BASE_URL}")
    print(f"  Auth:      {'X-API-Key set' if RAG_API_KEY else 'no auth (dev mode)'}")
    print(f"  DEMO_MODE: {demo_mode}")

    with httpx.Client(timeout=30.0) as client:
        # Health is always run first
        healthy = demo_health(client)

        if demo_mode in ("ingest", "all"):
            try:
                demo_ingest(client)
            except Exception as exc:
                print(f"\n  [ingest] Error: {exc}")

        if demo_mode in ("query", "all") and healthy:
            try:
                demo_query(client)
            except Exception as exc:
                print(f"\n  [query] Error: {exc}")

        if demo_mode in ("latency", "all") and healthy:
            try:
                demo_latency_comparison(client)
            except Exception as exc:
                print(f"\n  [latency] Error: {exc}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()
