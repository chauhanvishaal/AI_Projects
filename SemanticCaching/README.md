# Semantic Cache API

A distributed semantic caching service for AI agent workflows.  
Instead of repeating expensive LLM calls for semantically equivalent questions, this service stores and retrieves responses based on **cosine similarity** rather than exact string matching.

Fully free and self-hostable — no paid APIs, no cloud dependencies.

---

## Architecture

```
Client (agent / RAG / any HTTP consumer)
        │
        │  X-API-Key   HTTP/JSON
        ▼
┌──────────────────────────────────────────────┐
│                semcache-api                  │  FastAPI  :8000
│                                              │
│  ┌─────────────┐      ┌──────────────────┐  │
│  │   api.py    │─────▶│   semCache.py    │  │
│  │  (routes)   │      │  (cache engine)  │  │
│  └─────────────┘      └────────┬─────────┘  │
└───────────────────────────────-│─────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
        ┌──────────┐     ┌────────────┐    ┌───────────────┐
        │  Redis   │     │   FAISS    │    │    Ollama     │
        │  OSS 7   │     │  IndexIP   │    │  :11434       │
        │          │     │ (in-proc)  │    │               │
        │ • Store  │     │ • Cosine   │    │ • nomic-embed │
        │ • TTL    │     │   search   │    │   -text       │
        │ • Lock   │     │ • L2-norm  │    │   (embeddings)│
        │ • Stats  │     │   vectors  │    │               │
        │ • Pub/Sub│     └────────────┘    │ • llama3      │
        └──────────┘                       │   (LLM)       │
              │                            └───────────────┘
              │  Pub/Sub sync
              ▼
        [other semcache-api
          instances rebuild
          their FAISS index]
```

### Key design decisions

| Concern | Solution |
|---|---|
| **Semantic matching** | FAISS `IndexFlatIP` with L2-normalised vectors — inner product = cosine similarity, no external vector DB required |
| **Distributed state** | Redis is the canonical store; each node rebuilds FAISS from Redis on startup and syncs on writes via Pub/Sub |
| **Cache stampede** | Redis `SET NX EX` distributed lock + Lua atomic release + exponential back-off with re-check |
| **TTL & eviction** | Redis `EXPIRE` per entry; supports per-call TTL override |
| **Embeddings** | Ollama local server (`nomic-embed-text`, 768-dim) — no sentence-transformers or PyTorch required |
| **Auth** | `X-API-Key` header, `secrets.compare_digest` (constant-time, no timing attacks) |
| **LangChain** | `SemanticLLMCache(BaseCache)` — drop-in replacement for `set_llm_cache()`; all LLM calls are transparently cached |

### API routes

| Method | Path | Auth | Description |
|---|---|---|---|
| `POST` | `/query` | ✅ | Lookup or compute (lock-protected) and cache a response |
| `PUT` | `/cache` | ✅ | Store a pre-computed response (used by external services like RAG) |
| `DELETE` | `/invalidate/{key}` | ✅ | Delete a specific cache entry |
| `DELETE` | `/invalidate/agent/{agent_id}` | ✅ | Delete all entries for an agent |
| `GET` | `/stats/{agent_id}` | ✅ | Hit/miss statistics |
| `GET` | `/health` | ❌ | Redis + Ollama + FAISS status (unauthenticated) |

### Query flow

```
POST /query
  │
  ├─ embed(query) via Ollama nomic-embed-text
  ├─ FAISS search(k=1)           → cosine ≥ 0.92?
  │     HIT  → HGETALL from Redis → return response  (~5 ms)
  │     MISS ↓
  ├─ SET NX EX distributed lock  → prevent stampede
  ├─ re-check cache              → another node may have computed it
  ├─ call Ollama llama3          → generate response
  ├─ HSET + EXPIRE in Redis      → persist
  ├─ PUBLISH sync_channel        → notify peers to rebuild FAISS
  └─ Lua script atomic lock release
```

---

## Project structure

```
SemanticCaching/
├── semCache.py          # Core: SemanticCache, SemanticCacheConfig, SemanticLLMCache
├── api.py               # FastAPI HTTP layer
├── example_usage.py     # Demo: LangChain mode + HTTP API mode
├── requirements.txt     # Python dependencies
├── Dockerfile           # 2-stage build (builder → slim runtime)
├── docker-compose.yml   # Full stack: redis + ollama + semcache-api
├── .env.example         # All environment variables with documentation
└── .dockerignore
```

---

## Setup & install

### Option A — Docker Compose (recommended)

**Prerequisites:** Docker Desktop (or Docker Engine + Compose plugin)

```bash
cd d:\dev\py\AI_Projects\SemanticCaching

# 1. Create your .env file
cp .env.example .env

# 2. Generate a strong API key and paste it into .env
python -c "import secrets; print(secrets.token_urlsafe(32))"
# Edit .env:  API_KEY=<generated value>

# 3. Build and start all services
docker compose up --build

# Models are pulled automatically on first start (llama3 ~4 GB, nomic-embed-text ~270 MB).
# This can take several minutes on first run.

# 4. Verify
curl http://localhost:8000/health
```

Expected health response:
```json
{"redis": "ok", "ollama": "ok", "faiss_vectors": 0}
```

### Option B — Local development (no Docker)

**Prerequisites:** Python 3.12+, Redis, Ollama

```bash
# 1. Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# 2. Start Ollama and pull models
ollama serve
ollama pull llama3
ollama pull nomic-embed-text

# 3. Create and activate virtualenv
cd d:\dev\py\AI_Projects\SemanticCaching
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 4. Install dependencies
pip install -r requirements.txt

# 5. Set environment variables (or copy and edit .env, then load it)
set REDIS_URL=redis://localhost:6379
set OLLAMA_BASE_URL=http://localhost:11434
set API_KEY=dev-key

# 6. Start the API
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379` | Redis connection string |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3` | LLM model name |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model (768-dim) |
| `SIMILARITY_THRESHOLD` | `0.92` | Cosine similarity floor for a cache hit |
| `DEFAULT_TTL` | `3600` | Entry TTL in seconds |
| `NAMESPACE` | `default` | Redis key prefix — isolate per environment |
| `LOCK_TIMEOUT` | `30` | Distributed lock max hold time (seconds) |
| `MAX_LOCK_RETRIES` | `10` | Lock acquisition retries before giving up |
| `API_KEY` | *(empty)* | X-API-Key value; leave empty to disable auth (dev only) |

---

## Start / stop

```bash
# Start (detached)
docker compose up -d

# View logs
docker compose logs -f semcache-api

# Stop (keep volumes)
docker compose down

# Stop and wipe all data (Redis + Ollama models)
docker compose down -v
```

---

## Testing

### Quick smoke test (curl)

```bash
API_KEY="your-key-here"
BASE="http://localhost:8000"

# Health (no auth)
curl $BASE/health

# First query — cache miss, calls Ollama
curl -s -X POST $BASE/query \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?", "agent_id": "test"}' | python -m json.tool

# Similar query — should be a cache hit (cosine ≥ 0.92)
curl -s -X POST $BASE/query \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "Which city is the capital of France?", "agent_id": "test"}' | python -m json.tool

# Stats
curl -s $BASE/stats/test -H "X-API-Key: $API_KEY" | python -m json.tool

# Store a pre-computed response (used by the RAG service)
curl -s -X PUT $BASE/cache \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "How does FAISS work?", "response": "FAISS is a library for efficient similarity search.", "agent_id": "test"}' | python -m json.tool

# Invalidate a specific agent
curl -s -X DELETE $BASE/invalidate/agent/test -H "X-API-Key: $API_KEY"
```

### Example usage script

```bash
# Run both demo modes (LangChain in-process + HTTP API)
cd d:\dev\py\AI_Projects\SemanticCaching
set DEMO_MODE=both
set API_KEY=your-key-here
set API_BASE_URL=http://localhost:8000
python example_usage.py

# HTTP mode only
set DEMO_MODE=http
python example_usage.py
```

### LangChain in-process integration test

```python
from langchain.globals import set_llm_cache
from langchain_ollama import OllamaLLM
from semCache import SemanticCache, SemanticCacheConfig, SemanticLLMCache

cache = SemanticCache(SemanticCacheConfig())
set_llm_cache(SemanticLLMCache(cache))

llm = OllamaLLM(model="llama3")

r1 = llm.invoke("What is quantum entanglement?")   # miss — calls Ollama
r2 = llm.invoke("Explain quantum entanglement.")   # hit  — from cache
assert r1 == r2
print("Cache hit confirmed ✓")
```

### Verify distributed sync

Start two API instances and observe FAISS sync via Redis Pub/Sub:

```bash
# Terminal 1
uvicorn api:app --port 8000

# Terminal 2
uvicorn api:app --port 8001

# Write to instance 1
curl -s -X POST http://localhost:8000/query \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"query": "test sync", "agent_id": "sync-test"}'

# Read similar query from instance 2 — should be a cache hit
curl -s -X POST http://localhost:8001/query \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"query": "testing sync", "agent_id": "sync-test"}'
```

---

## Similarity threshold tuning

| Use case | Recommended threshold |
|---|---|
| Exact or near-exact match only | `0.97` |
| Default — captures paraphrases | `0.92` |
| Aggressive caching, tolerates drift | `0.85` |

Change via `SIMILARITY_THRESHOLD` in `.env` (no restart required if set before startup).
