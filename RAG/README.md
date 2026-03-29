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
│  app.py  (lifespan + router wiring)                      │
│    │                                                     │
│    ├── routers/query.py    POST /query                   │
│    │     │  1. dependencies.get_cache_client().lookup()  │
│    │     │        HIT  → return instantly (~5 ms)        │
│    │     │        MISS ↓                                 │
│    │     │  2. rag_chain.run(question, top_k)            │
│    │     │        store.vector_store_manager             │
│    │     │          → FAISS retriever                    │
│    │     │          → ChatPromptTemplate                 │
│    │     │          → models.get_llm() (Ollama)         │
│    │     │          → StrOutputParser                    │
│    │     │  3. cache_client.store()  (fire-and-forget)   │
│    │     │                                               │
│    ├── routers/ingest.py   POST /ingest  DELETE /index   │
│    │     └─ ingestion_service.ingest(file_paths)         │
│    │             → DocumentIngestionService              │
│    │               (load → split → embed → FAISS)        │
│    │                                                     │
│    └── routers/health.py   GET /health  (no auth)        │
│                                                          │
│  auth.py          X-API-Key  secrets.compare_digest      │
│  config.py        Typed settings (frozen dataclass)      │
│  schemas.py       Pydantic request / response models     │
└──────────────────────────────────────────────────────────┘
              │ HTTP                │ HTTP
              ▼                    ▼
   ┌────────────────────┐  ┌───────────────┐
   │   semcache-api     │  │    Ollama     │
   │   :8000            │  │   :11434      │
   │  POST /query       │  │ • llama3      │
   │  PUT  /cache       │  │ • nomic-embed │
   └────────┬───────────┘  └───────┬───────┘
            │                      │ embeddings
            ▼                      ▼
      ┌──────────┐      ┌─────────────────────┐
      │  Redis   │      │  FAISS index        │
      │  OSS 7   │      │  (disk-persisted,   │
      └──────────┘      │   Docker volume)    │
                        └─────────────────────┘
```

---

## Module design — Single Responsibility Principle

Each module has **exactly one reason to change**:

| Module | Responsibility | Changes when… |
|---|---|---|
| `config.py` | Typed settings from env vars (`frozen` dataclass) | A new env var or default is added |
| `schemas.py` | Pydantic request / response models | API contract changes |
| `models.py` | Ollama LLM + embeddings singletons | Switching model provider |
| `store.py` | `VectorStoreManager` — FAISS lifecycle (load, save, clear) | Switching vector database |
| `ingestion.py` | `DocumentIngestionService` — load, chunk, embed, index | New file formats or splitter strategy |
| `rag_chain.py` | `RAGChain` — FAISS retrieval + Ollama LLM generation | Prompt, retrieval strategy, or LLM changes |
| `auth.py` | `require_api_key` FastAPI dependency | Switching auth mechanism |
| `dependencies.py` | Shared injectable FastAPI dependencies | New app-wide singletons |
| `routers/query.py` | `POST /query` orchestration | Query endpoint behaviour |
| `routers/ingest.py` | `POST /ingest`, `DELETE /index` | Upload / index management |
| `routers/health.py` | `GET /health` | Health probe logic |
| `app.py` | FastAPI factory, lifespan, router registration | Adding/removing routers |
| `cache_client.py` | HTTP gateway to `semcache-api` | Cache service API changes |
| `workflow.py` | Backward-compat shim (`from app import app`) | Deprecated |

### Query flow

```
POST /query  →  routers/query.py
  │
  1. dependencies.get_cache_client().lookup(question)
  │         HIT  → return instantly  (cache_hit=True, sources=[], chunks=0)
  │         MISS ↓
  │
  2. rag_chain.run(question, top_k)           ← RAGChain (rag_chain.py)
  │         store.vector_store_manager        ← VectorStoreManager (store.py)
  │           → FAISS.as_retriever(k=top_k)
  │           → _format_docs(chunks)
  │           → ChatPromptTemplate            ← prompt in rag_chain.py
  │           → models.get_llm()             ← OllamaLLM (models.py)
  │           → StrOutputParser
  │
  3. cache_client.store(question, answer)
              → PUT semcache-api:8000/cache   (fire-and-forget)
```

### Supported document types

| Extension | Loader |
|---|---|
| `.pdf` | `PyPDFLoader` |
| `.md`, `.markdown` | `UnstructuredMarkdownLoader` |
| `.txt`, `.text` | `TextLoader` |

### API routes

| Method | Path | Auth | Module | Description |
|---|---|---|---|---|
| `POST` | `/query` | ✅ | `routers/query.py` | Answer a question via RAG (with semantic cache) |
| `POST` | `/ingest` | ✅ | `routers/ingest.py` | Upload documents (multipart/form-data) |
| `DELETE` | `/index` | ✅ | `routers/ingest.py` | Wipe the FAISS index |
| `GET` | `/health` | ❌ | `routers/health.py` | Ollama + cache + FAISS status |

`POST /query` accepts `use_cache: false` to force a fresh RAG call, bypassing the semantic cache.

---

## Project structure

```
RAG/
├── app.py               # FastAPI factory — lifespan + router wiring
├── config.py            # Typed settings (frozen dataclass, env vars)
├── schemas.py           # Pydantic request / response models
├── models.py            # Ollama LLM + embeddings singletons
├── store.py             # VectorStoreManager — FAISS persistence lifecycle
├── ingestion.py         # DocumentIngestionService — load, chunk, embed, index
├── rag_chain.py         # RAGChain — FAISS retrieval + Ollama LLM generation
├── auth.py              # X-API-Key FastAPI dependency
├── dependencies.py      # Shared injectable dependencies
├── cache_client.py      # HTTP gateway to semcache-api (SRP: cache only)
├── routers/
│   ├── __init__.py
│   ├── query.py         # POST /query
│   ├── ingest.py        # POST /ingest, DELETE /index
│   └── health.py        # GET /health
├── workflow.py          # Backward-compat shim → from app import app
├── example_usage.py     # Demo: health, ingest, query, latency comparison
├── requirements.txt     # Python dependencies
├── Dockerfile           # 2-stage build (builder → slim runtime)
├── docker-compose.yml   # Full stack: ollama + redis + semcache-api + rag-api
├── .env.example         # All environment variables with documentation
└── .dockerignore
```

---

## Build & test — step by step

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows / macOS) or Docker Engine + Compose plugin (Linux)
- `curl` (or any HTTP client)
- Python 3.12+ (only needed for local dev / unit tests)

---

### Step 1 — Clone / navigate to the project

```bash
cd d:\dev\py\AI_Projects\RAG
```

---

### Step 2 — Create environment file

```bash
cp .env.example .env
```

Generate two strong API keys — one for `rag-api`, one for `semcache-api`:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
# run twice — copy the two outputs
```

Open `.env` and set:

```ini
RAG_API_KEY=<first key>
SEMCACHE_API_KEY=<second key>
```

---

### Step 3 — Build all Docker images

```bash
docker compose build
```

This builds:
- `rag-api` from `RAG/Dockerfile` (2-stage, `python:3.12-slim`)
- `semcache-api` from `../SemanticCaching/Dockerfile`

---

### Step 4 — Start the full stack

```bash
docker compose up -d
```

Services started and their ports:

| Service | Host port | Purpose |
|---|---|---|
| `ollama` | 11434 | LLM + embedding model server |
| `ollama-init` | — | One-shot: pulls `llama3` and `nomic-embed-text` |
| `redis` | 6379 | Semantic cache store |
| `semcache-api` | 8000 | Semantic Cache REST API |
| `rag-api` | 8001 | RAG Workflow REST API |

> **First run:** `ollama-init` pulls `llama3` (~4 GB) and `nomic-embed-text` (~270 MB). This happens once; subsequent starts use the cached `ollama-data` volume. `rag-api` will not start until model pulls complete.

Watch startup progress:

```bash
docker compose logs -f
```

---

### Step 5 — Verify health

```bash
# RAG service (unauthenticated)
curl -s http://localhost:8001/health | python -m json.tool
```

Expected:
```json
{
  "ollama": "ok",
  "semantic_cache": "ok",
  "faiss_vectors": 0,
  "docs_path": "/data/documents"
}
```

```bash
# Semantic cache service
curl -s http://localhost:8000/health | python -m json.tool
```

Expected:
```json
{"redis": "ok", "ollama": "ok", "faiss_vectors": 0}
```

If either service shows `"fail"`, check `docker compose logs <service-name>`.

---

### Step 6 — Ingest documents

Upload a document so the RAG pipeline has something to retrieve from:

```bash
RAG_KEY="<your RAG_API_KEY from .env>"

# Using the sample text from example_usage.py (auto-created if you run the script)
# Or upload your own:
curl -s -X POST http://localhost:8001/ingest \
  -H "X-API-Key: $RAG_KEY" \
  -F "files=@./sample_docs/sample.txt" \
  | python -m json.tool
```

Expected:
```json
{
  "chunks_added": 3,
  "total_vectors": 3,
  "latency_ms": 812.4
}
```

---

### Step 7 — Run a query (cache miss → RAG)

```bash
curl -s -X POST http://localhost:8001/query \
  -H "X-API-Key: $RAG_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Retrieval-Augmented Generation?", "top_k": 4}' \
  | python -m json.tool
```

Expected (first call — cache miss, full RAG chain):
```json
{
  "answer": "Retrieval-Augmented Generation (RAG) combines ...",
  "sources": ["sample.txt"],
  "chunks_retrieved": 3,
  "cache_hit": false,
  "latency_ms": 4231.0
}
```

---

### Step 8 — Run a semantically similar query (cache hit)

```bash
curl -s -X POST http://localhost:8001/query \
  -H "X-API-Key: $RAG_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "Can you explain what RAG means in AI?", "top_k": 4}' \
  | python -m json.tool
```

Expected (similar question — cosine ≥ 0.92 → cache hit, ~ms latency):
```json
{
  "answer": "Retrieval-Augmented Generation (RAG) combines ...",
  "sources": [],
  "chunks_retrieved": 0,
  "cache_hit": true,
  "latency_ms": 47.2
}
```

---

### Step 9 — Run the full demo script

```bash
# Requires the services to be running (Step 4)
set RAG_API_KEY=<your key>          # Windows
# export RAG_API_KEY=<your key>     # macOS/Linux

python example_usage.py
```

This runs all four demo sections automatically:
1. Health check
2. Document ingestion (auto-creates `./sample_docs/sample.txt` if needed)
3. Query with cache miss then hit
4. Latency comparison (shows typical 80× speed-up)

---

### Step 10 — Unit test: cache client graceful degradation

This test runs without Docker — it verifies `cache_client.py` behaves safely when `semcache-api` is unreachable:

```bash
cd d:\dev\py\AI_Projects\RAG

# Windows (set env vars before running)
set SEMCACHE_BASE_URL=http://localhost:9999
set SEMCACHE_TIMEOUT=1.0
python -c "
from cache_client import SemanticCacheClient
c = SemanticCacheClient()
assert c.is_available() is False, 'Expected False'
assert c.lookup('any question') is None, 'Expected None'
c.store('any question', 'any answer')
print('cache_client graceful degradation: PASS')
"
```

---

### Step 11 — Test auth enforcement

```bash
# Missing API key — expect 403
curl -s -o /dev/null -w "%{http_code}" \
  -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}'
# Expected: 403

# Wrong key — expect 403
curl -s -o /dev/null -w "%{http_code}" \
  -X POST http://localhost:8001/query \
  -H "X-API-Key: wrong-key" \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}'
# Expected: 403

# Correct key — expect 200 or 503 (503 = no documents ingested yet, not an auth failure)
curl -s -o /dev/null -w "%{http_code}" \
  -X POST http://localhost:8001/query \
  -H "X-API-Key: $RAG_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "test"}'
# Expected: 200 (if documents ingested) or 503 (if not)
```

---

### Step 12 — Test index management

```bash
# Clear the FAISS index
curl -s -X DELETE http://localhost:8001/index \
  -H "X-API-Key: $RAG_KEY" | python -m json.tool
# Expected: {"cleared": true}

# Confirm 0 vectors in health
curl -s http://localhost:8001/health | python -m json.tool
# Expected: "faiss_vectors": 0

# Query with no documents → expect 503
curl -s -X POST http://localhost:8001/query \
  -H "X-API-Key: $RAG_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "anything"}' | python -m json.tool
# Expected: 503 with detail message
```

---

### Stop the stack

```bash
# Stop services, keep data volumes (FAISS index + Ollama models survive)
docker compose down

# Stop and wipe all data (models will re-download on next start)
docker compose down -v
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | LLM model for answer generation |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model (must match `semcache`) |
| `FAISS_INDEX_PATH` | `/data/faiss_index` | Disk path for persisted FAISS index |
| `DOCS_PATH` | `/data/documents` | Documents auto-ingested on startup |
| `CHUNK_SIZE` | `512` | Characters per text chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between adjacent chunks |
| `RETRIEVAL_TOP_K` | `4` | Chunks retrieved per query |
| `API_KEY` | *(empty)* | X-API-Key for RAG API; empty = no auth (dev only) |
| `SEMCACHE_BASE_URL` | `http://semcache-api:8000` | URL of the Semantic Cache service |
| `SEMCACHE_API_KEY` | *(empty)* | Must match `API_KEY` in the cache service |
| `SEMCACHE_AGENT_ID` | `rag-workflow` | Namespaces RAG cache entries |
| `SEMCACHE_TTL` | `3600` | How long RAG answers stay in the cache (seconds) |
| `SEMCACHE_TIMEOUT` | `5.0` | Max wait for cache HTTP calls (seconds) |

---

## Local development (without Docker)

```bash
# 1. Start the Semantic Cache service
cd d:\dev\py\AI_Projects\SemanticCaching
docker compose up -d          # semcache-api on :8000

# 2. Set up RAG virtualenv
cd d:\dev\py\AI_Projects\RAG
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -r requirements.txt

# 3. Set env vars (Windows)
set OLLAMA_BASE_URL=http://localhost:11434
set OLLAMA_MODEL=llama3
set EMBEDDING_MODEL=nomic-embed-text
set FAISS_INDEX_PATH=./data/faiss_index
set DOCS_PATH=./data/documents
set API_KEY=dev-key
set SEMCACHE_BASE_URL=http://localhost:8000
set SEMCACHE_API_KEY=<semcache API_KEY value>
set SEMCACHE_AGENT_ID=rag-workflow

# 4. Start with live reload
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

> `workflow:app` still works as a backward-compatible alias but `app:app` is preferred.

---

## Relationship to Semantic Cache service

| Aspect | `semcache-api` | `rag-api` |
|---|---|---|
| Port | 8000 | 8001 |
| Responsibility | Semantic cache (FAISS + Redis + Ollama embeddings) | RAG pipeline (document ingestion + retrieval + generation) |
| Knows about the other? | No | No (delegates via HTTP) |
| Can be used standalone? | Yes — any HTTP client | Yes — degrades gracefully if semcache-api is down |
| Scale independently? | Yes | Yes |

See [SemanticCaching/README.md](../SemanticCaching/README.md) for full cache service documentation.


