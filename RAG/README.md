# RAG Workflow API

A containerised Retrieval-Augmented Generation (RAG) pipeline that answers questions grounded in your own documents — using a fully free, self-hosted stack.

Semantic caching is provided by the companion **Semantic Cache** service ([`../SemanticCaching`](../SemanticCaching/README.md)). The two services follow the **Single Responsibility Principle**: this service knows only about RAG; it delegates all caching concerns over HTTP to `semcache-api`.

---

## Architecture

```
Client
  │  X-API-Key   HTTP/JSON
  ▼
┌──────────────────────────────────────────────────────────┐
│                      rag-api  :8001                      │
│                                                          │
│  POST /query                                             │
│    │                                                     │
│    ├─ 1. cache_client.lookup(question)   ─────────────┐  │
│    │         HIT  → return instantly (~5 ms)          │  │
│    │         MISS ↓                                   │  │
│    ├─ 2. _run_rag(question, top_k)                    │  │
│    │         FAISS retriever                          │  │
│    │           → format chunks                        │  │
│    │           → ChatPromptTemplate                   │  │
│    │           → OllamaLLM                            │  │
│    │           → StrOutputParser                      │  │
│    │                                                  │  │
│    └─ 3. cache_client.store(question, answer) ───────►│  │
│                                                       │  │
│  cache_client.py ─────────── HTTP ───────────────────►│  │
└───────────────────────────────────────────────────────┼──┘
                                                        │
                    ┌───────────────────────────────────┘
                    ▼
        ┌────────────────────┐      ┌───────────────┐
        │   semcache-api     │      │    Ollama     │
        │   :8000            │      │   :11434      │
        │                    │      │               │
        │  POST /query       │      │ • llama3 (LLM)│
        │  PUT  /cache       │      │ • nomic-embed │
        │                    │      │   -text       │
        └────────┬───────────┘      └───────┬───────┘
                 │                          │
                 ▼                          │ embeddings
           ┌──────────┐         ┌───────────┴─────────┐
           │  Redis   │         │     FAISS index      │
           │  OSS 7   │         │  (disk-persisted,    │
           │          │         │   Docker volume)     │
           └──────────┘         └─────────────────────┘
```

### Single Responsibility design

Each file has exactly **one reason to change:**

| File | Responsibility |
|---|---|
| `workflow.py` | RAG pipeline only (load, chunk, embed, retrieve, generate) |
| `cache_client.py` | HTTP gateway to `semcache-api` only — no RAG logic |

`workflow.py` calls `cache_client.lookup()` and `cache_client.store()` but has zero knowledge of Redis, FAISS vectors, or similarity thresholds — those are the cache service's concern.

### Query flow

```
POST /query  (rag-api:8001)
  │
  1. SemanticCacheClient.lookup(question)
  │         → POST semcache-api:8000/query
  │         HIT  → return instantly  (cache_hit=True, sources=[], chunks=0)
  │         MISS ↓
  │
  2. _run_rag(question, top_k)              ← this service's only real job
  │         FAISS.as_retriever(k=top_k)
  │           → retrieve chunks
  │           → _format_docs([source, content])
  │           → ChatPromptTemplate.from_template(...)
  │           → OllamaLLM(llama3)
  │           → StrOutputParser()
  │
  3. SemanticCacheClient.store(question, answer)
              → PUT semcache-api:8000/cache
              (fire-and-forget — never blocks the response)
```

### Supported document types

| Extension | Loader |
|---|---|
| `.pdf` | `PyPDFLoader` |
| `.md`, `.markdown` | `UnstructuredMarkdownLoader` |
| `.txt`, `.text` | `TextLoader` |

### API routes

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/query` | ✅ | Answer a question via RAG (with semantic cache) |
| `POST` | `/ingest` | ✅ | Upload documents (multipart/form-data) |
| `DELETE` | `/index` | ✅ | Wipe the FAISS index |
| `GET` | `/health` | ❌ | Ollama + cache + FAISS status (unauthenticated) |

`POST /query` accepts `use_cache: false` to force a fresh RAG call, bypassing the semantic cache entirely.

---

## Project structure

```
RAG/
├── workflow.py          # FastAPI app — RAG pipeline (ingestion + retrieval + LLM)
├── cache_client.py      # SRP HTTP client for semcache-api
├── example_usage.py     # Demo: health, ingest, query, latency comparison
├── requirements.txt     # Python dependencies
├── Dockerfile           # 2-stage build (builder → slim runtime)
├── docker-compose.yml   # Full stack: ollama + redis + semcache-api + rag-api
├── .env.example         # All environment variables with documentation
└── .dockerignore
```

---

## Setup & install

### Option A — Docker Compose (recommended)

This brings up the **complete stack**: Ollama, Redis, semcache-api, and rag-api together.

**Prerequisites:** Docker Desktop (or Docker Engine + Compose plugin)

```bash
cd d:\dev\py\AI_Projects\RAG

# 1. Create your .env file
cp .env.example .env

# 2. Generate strong API keys (one for RAG, one for the cache service)
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Edit .env and set:
#   RAG_API_KEY=<first generated value>
#   SEMCACHE_API_KEY=<second generated value>

# 3. Build and start all services
docker compose up --build

# First run pulls Ollama models (llama3 ~4 GB, nomic-embed-text ~270 MB).
# semcache-api starts on :8000, rag-api on :8001.

# 4. Verify
curl http://localhost:8001/health
```

Expected health response:
```json
{
  "ollama": "ok",
  "semantic_cache": "ok",
  "faiss_vectors": 0,
  "docs_path": "/data/documents"
}
```

### Option B — Local development (no Docker for RAG, cache in Docker)

```bash
# 1. Start the Semantic Cache service (prerequisite)
cd d:\dev\py\AI_Projects\SemanticCaching
docker compose up -d          # exposes semcache-api on :8000

# 2. Set up RAG virtualenv
cd d:\dev\py\AI_Projects\RAG
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt

# 3. Configure environment
set OLLAMA_BASE_URL=http://localhost:11434
set EMBEDDING_MODEL=nomic-embed-text
set OLLAMA_MODEL=llama3
set FAISS_INDEX_PATH=./data/faiss_index
set DOCS_PATH=./data/documents
set API_KEY=dev-key
set SEMCACHE_BASE_URL=http://localhost:8000
set SEMCACHE_API_KEY=<your semcache API_KEY>
set SEMCACHE_AGENT_ID=rag-workflow

# 4. Start the RAG API
uvicorn workflow:app --host 0.0.0.0 --port 8001 --reload
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | LLM model for answer generation |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model (must match semcache) |
| `FAISS_INDEX_PATH` | `/data/faiss_index` | Disk location for persisted FAISS index |
| `DOCS_PATH` | `/data/documents` | Documents auto-ingested on startup |
| `CHUNK_SIZE` | `512` | Token chunk size for text splitting |
| `CHUNK_OVERLAP` | `64` | Overlap between adjacent chunks |
| `RETRIEVAL_TOP_K` | `4` | Number of chunks retrieved per query |
| `API_KEY` | *(empty)* | X-API-Key for RAG API; empty = no auth (dev only) |
| `SEMCACHE_BASE_URL` | `http://semcache-api:8000` | URL of the Semantic Cache service |
| `SEMCACHE_API_KEY` | *(empty)* | Must match `API_KEY` in the cache service |
| `SEMCACHE_AGENT_ID` | `rag-workflow` | Namespaces RAG cache entries |
| `SEMCACHE_TTL` | `3600` | How long RAG answers stay in the cache (seconds) |
| `SEMCACHE_TIMEOUT` | `5.0` | Max seconds to wait for cache HTTP calls |

---

## Start / stop

```bash
# Start (detached)
docker compose up -d

# View logs for just the RAG service
docker compose logs -f rag-api

# View logs for all services
docker compose logs -f

# Stop (keep volumes / FAISS index)
docker compose down

# Stop and wipe all data
docker compose down -v
```

---

## Testing

### Quick smoke test (curl)

```bash
RAG_KEY="your-rag-api-key"
BASE="http://localhost:8001"

# Health (no auth)
curl $BASE/health

# Ingest a document
curl -s -X POST $BASE/ingest \
  -H "X-API-Key: $RAG_KEY" \
  -F "files=@./sample_docs/sample.txt" | python -m json.tool

# First query — cache miss, full RAG chain runs
curl -s -X POST $BASE/query \
  -H "X-API-Key: $RAG_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Retrieval-Augmented Generation?", "top_k": 4}' \
  | python -m json.tool

# Semantically similar query — should be a cache hit
curl -s -X POST $BASE/query \
  -H "X-API-Key: $RAG_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "Can you explain what RAG means in AI?", "top_k": 4}' \
  | python -m json.tool

# Force a fresh RAG call (bypass cache)
curl -s -X POST $BASE/query \
  -H "X-API-Key: $RAG_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "use_cache": false}' \
  | python -m json.tool
```

### Example usage script

The included `example_usage.py` runs four demo sections automatically:

```bash
cd d:\dev\py\AI_Projects\RAG

# Run all sections (health → ingest → query → latency comparison)
RAG_API_KEY=your-key python example_usage.py

# Individual sections
DEMO_MODE=ingest  RAG_API_KEY=your-key python example_usage.py
DEMO_MODE=query   RAG_API_KEY=your-key python example_usage.py
DEMO_MODE=latency RAG_API_KEY=your-key python example_usage.py
```

The script auto-creates a `./sample_docs/sample.txt` with RAG-related content if no documents are found, so it runs end-to-end without requiring real files.

Expected output pattern:
```
1. Health check — GET /health
  Ollama:         ok
  Semantic cache: ok
  FAISS vectors:  0

2. Document ingestion — POST /ingest
  Uploading 1 file(s): sample.txt
  Chunks added:   3
  Total vectors:  3

3. Querying — POST /query
  Question (cache-enabled): 'What is Retrieval-Augmented Generation?'
  Cache: MISS (RAG chain executed)
  Chunks retrieved: 3
  Latency: 4231 ms

  Question (cache-enabled): 'Can you explain what RAG means in AI?'
  Cache: HIT  (returned from semantic cache)
  Chunks retrieved: 0
  Latency: 48 ms

4. Latency comparison — cache vs RAG
  RAG (first call, cache miss):  MISS | latency: 4 s
  Cache hit (same question):     HIT  | latency: 50 ms
  Speed-up (miss → hit):         80×
```

### Unit-test cache isolation

Verify that the `cache_client` degrades gracefully when the cache service is unavailable:

```python
# test_cache_client.py
import os
os.environ["SEMCACHE_BASE_URL"] = "http://localhost:9999"  # unreachable
os.environ["SEMCACHE_TIMEOUT"] = "1.0"

from cache_client import SemanticCacheClient

client = SemanticCacheClient()

# is_available() should return False, not raise
assert client.is_available() is False

# lookup() should return None (graceful miss), not raise
assert client.lookup("any question") is None

# store() should be silent, not raise
client.store("any question", "any answer")

print("Graceful degradation confirmed ✓")
```

Run it:
```bash
python test_cache_client.py
```

### End-to-end integration test

With both services running:

```bash
RAG_KEY="your-rag-key"

# 1. Ingest
curl -s -X POST http://localhost:8001/ingest \
  -H "X-API-Key: $RAG_KEY" \
  -F "files=@./sample_docs/sample.txt"

# 2. First query (cache miss)
RESP=$(curl -s -X POST http://localhost:8001/query \
  -H "X-API-Key: $RAG_KEY" -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}')
echo "cache_hit: $(echo $RESP | python -m json.tool | grep cache_hit)"

# 3. Same query again (should be cache hit)
RESP=$(curl -s -X POST http://localhost:8001/query \
  -H "X-API-Key: $RAG_KEY" -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}')
echo "cache_hit: $(echo $RESP | python -m json.tool | grep cache_hit)"
# Expected: "cache_hit": true
```

---

## Relationship to Semantic Cache service

| Aspect | semcache-api | rag-api |
|---|---|---|
| Port | 8000 | 8001 |
| Responsibility | Semantic cache (FAISS + Redis + Ollama embeddings) | RAG pipeline (document ingestion + retrieval + LLM generation) |
| Knows about RAG? | No | No (delegates via HTTP) |
| Knows about Redis/FAISS vectors? | Yes | No |
| Can be used standalone? | Yes — any HTTP client | Yes — degrades gracefully if semcache-api is down |
| Scale independently? | Yes | Yes |

See [SemanticCaching/README.md](../SemanticCaching/README.md) for full cache service documentation.
