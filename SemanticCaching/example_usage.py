"""
example_usage.py — Demonstrates both integration modes
=======================================================

MODE 1: In-process LangChain path
  SemanticLLMCache is registered with LangChain.
  Every LLM call is transparently routed through the semantic cache.

MODE 2: HTTP API path
  A downstream agent calls the REST API with X-API-Key authentication.
  Suitable for any language / framework.

Prerequisites (local development):
  1. Redis running:  docker run -d -p 6379:6379 redis:7-alpine
  2. Ollama running: ollama serve  (and: ollama pull llama3)
  3. API server:     uvicorn api:app --port 8000  (for Mode 2)
  4. .env loaded (or env vars set)
"""

import os
import time

# ---------------------------------------------------------------------------
# MODE 1 — In-process LangChain cache
# ---------------------------------------------------------------------------

def demo_langchain_mode() -> None:
    print("\n" + "=" * 60)
    print("MODE 1 — In-process LangChain SemanticLLMCache")
    print("=" * 60)

    from langchain.globals import set_llm_cache
    from langchain_ollama import OllamaLLM

    from semCache import SemanticCache, SemanticCacheConfig, SemanticLLMCache

    config = SemanticCacheConfig()
    cache = SemanticCache(config)
    set_llm_cache(SemanticLLMCache(cache))

    llm = OllamaLLM(
        model=os.environ.get("OLLAMA_MODEL", "llama3"),
        base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    )

    query = "What is the capital of France?"

    # First call — cache miss, hits Ollama
    print(f"\nQuery: {query}")
    t0 = time.perf_counter()
    response1 = llm.invoke(query)
    t1 = time.perf_counter()
    print(f"Response (miss): {response1[:120]}...")
    print(f"Latency (miss):  {(t1 - t0) * 1000:.0f} ms")

    # Second call — semantically similar query, should hit cache
    similar_query = "Which city is the capital of France?"
    print(f"\nQuery (similar): {similar_query}")
    t0 = time.perf_counter()
    response2 = llm.invoke(similar_query)
    t1 = time.perf_counter()
    print(f"Response (hit):  {response2[:120]}...")
    print(f"Latency (hit):   {(t1 - t0) * 1000:.0f} ms")

    # Stats
    stats = cache.get_stats("lc-" + __import__("hashlib").sha256(llm.model.encode()).hexdigest()[:12])
    print(f"\nStats: {stats}")


# ---------------------------------------------------------------------------
# MODE 2 — HTTP API path (requires api.py server running on :8000)
# ---------------------------------------------------------------------------

def demo_http_api_mode() -> None:
    print("\n" + "=" * 60)
    print("MODE 2 — HTTP API client")
    print("=" * 60)

    import httpx

    api_base = os.environ.get("API_BASE_URL", "http://localhost:8000")
    api_key = os.environ.get("API_KEY", "")
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    # Health check (no auth required)
    health = httpx.get(f"{api_base}/health").json()
    print(f"\n/health: {health}")

    query = "Explain quantum entanglement in simple terms"
    agent_id = "demo-agent"

    # First call
    print(f"\nPOST /query (first call — expect miss)")
    t0 = time.perf_counter()
    resp = httpx.post(
        f"{api_base}/query",
        headers=headers,
        json={"query": query, "agent_id": agent_id},
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()
    t1 = time.perf_counter()
    print(f"  cache_hit:  {data['cache_hit']}")
    print(f"  latency:    {data['latency_ms']} ms")
    print(f"  response:   {data['response'][:120]}...")

    # Second call — semantically similar
    similar = "Can you explain quantum entanglement simply?"
    print(f"\nPOST /query (similar query — expect hit)")
    t0 = time.perf_counter()
    resp = httpx.post(
        f"{api_base}/query",
        headers=headers,
        json={"query": similar, "agent_id": agent_id},
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()
    t1 = time.perf_counter()
    print(f"  cache_hit:  {data['cache_hit']}")
    print(f"  latency:    {data['latency_ms']} ms")
    print(f"  response:   {data['response'][:120]}...")

    # Stats
    resp = httpx.get(f"{api_base}/stats/{agent_id}", headers=headers)
    resp.raise_for_status()
    print(f"\n/stats/{agent_id}: {resp.json()}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mode = os.environ.get("DEMO_MODE", "both").lower()

    if mode in ("langchain", "both"):
        try:
            demo_langchain_mode()
        except Exception as exc:
            print(f"[LangChain mode] Error: {exc}")

    if mode in ("http", "both"):
        try:
            demo_http_api_mode()
        except Exception as exc:
            print(f"[HTTP API mode] Error: {exc}")
