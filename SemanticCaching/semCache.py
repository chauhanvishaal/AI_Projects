"""
semCache.py — Semantic Cache for AI Agents
==========================================
Distributed semantic caching using:
  - FAISS (local in-process vector search, cosine similarity)
  - Redis OSS (canonical store, distributed lock, TTL, Pub/Sub sync, stats)
  - Ollama embeddings API (local, free — same container as the LLM)
  - LangChain BaseCache integration (SemanticLLMCache)

All components are 100% free and self-hostable.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import httpx
import faiss
import numpy as np
import redis
from langchain.schema import Generation
from langchain_core.caches import BaseCache

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SemanticCacheConfig:
    """All settings for SemanticCache, readable from environment variables."""

    redis_url: str = field(
        default_factory=lambda: os.environ.get("REDIS_URL", "redis://localhost:6379")
    )
    embedding_model: str = field(
        default_factory=lambda: os.environ.get(
            "EMBEDDING_MODEL", "nomic-embed-text"
        )
    )
    ollama_base_url: str = field(
        default_factory=lambda: os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
    )
    similarity_threshold: float = field(
        default_factory=lambda: float(
            os.environ.get("SIMILARITY_THRESHOLD", "0.92")
        )
    )
    default_ttl: int = field(
        default_factory=lambda: int(os.environ.get("DEFAULT_TTL", "3600"))
    )
    lock_timeout: int = field(
        default_factory=lambda: int(os.environ.get("LOCK_TIMEOUT", "30"))
    )
    max_lock_retries: int = field(
        default_factory=lambda: int(os.environ.get("MAX_LOCK_RETRIES", "10"))
    )
    namespace: str = field(
        default_factory=lambda: os.environ.get("NAMESPACE", "default")
    )

    # Key patterns (derived)
    @property
    def key_prefix(self) -> str:
        return f"semcache:{self.namespace}"

    @property
    def sync_channel(self) -> str:
        return f"{self.key_prefix}:sync"

    @property
    def invalidate_channel(self) -> str:
        return f"{self.key_prefix}:invalidate"


# ---------------------------------------------------------------------------
# Embedding wrapper — uses Ollama /api/embeddings (no extra model downloads)
# ---------------------------------------------------------------------------

class _EmbeddingModel:
    """Calls the Ollama embeddings endpoint to produce L2-normalised float32
    vectors for cosine similarity via inner product in FAISS."""

    def __init__(self, model_name: str, ollama_base_url: str) -> None:
        self._model_name = model_name
        self._url = f"{ollama_base_url.rstrip('/')}/api/embeddings"
        logger.info("Probing Ollama embedding model '%s' at %s", model_name, self._url)
        # Probe on init to discover vector dimension and validate connectivity
        probe = self._embed_raw("probe")
        self.dim: int = len(probe)
        logger.info("Embedding model ready — dim=%d", self.dim)

    def _embed_raw(self, text: str) -> list[float]:
        """Call Ollama and return the raw embedding list."""
        try:
            resp = httpx.post(
                self._url,
                json={"model": self._model_name, "prompt": text},
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ollama embeddings error ({exc.response.status_code}): {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(f"Cannot reach Ollama at {self._url}: {exc}") from exc

    def encode(self, text: str) -> np.ndarray:
        """Return a 1-D L2-normalised float32 numpy array."""
        vec = np.array(self._embed_raw(text), dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


# ---------------------------------------------------------------------------
# SemanticCache — core class
# ---------------------------------------------------------------------------

class SemanticCache:
    """
    Distributed semantic cache backed by FAISS (local) and Redis (canonical).

    Thread-safety: public methods are protected by a reentrant lock.
    Distributed sync: peers receive new entries via Redis Pub/Sub so their
    local FAISS index stays consistent without full rebuilds at runtime.
    """

    def __init__(self, config: Optional[SemanticCacheConfig] = None) -> None:
        self.cfg = config or SemanticCacheConfig()
        self._lock = threading.RLock()

        # Redis
        self._redis = redis.Redis.from_url(
            self.cfg.redis_url,
            decode_responses=False,  # we store raw bytes for embeddings
        )

        # Embedding model — backed by Ollama (no separate model download)
        self._embedder = _EmbeddingModel(self.cfg.embedding_model, self.cfg.ollama_base_url)

        # FAISS — IndexFlatIP for exact cosine (vectors are L2-normalised)
        # Wrapped in IndexIDMap2 so we can use integer IDs we control
        self._flat_index = faiss.IndexFlatIP(self._embedder.dim)
        self._index = faiss.IndexIDMap2(self._flat_index)

        # ID mapping: faiss integer id → redis key (str)
        self._id_to_key: dict[int, str] = {}
        self._key_to_id: dict[str, int] = {}
        self._next_id: int = 0

        # Bootstrap FAISS from existing Redis entries
        self._rebuild_index()

        # Start Pub/Sub listener for peer sync
        self._start_pubsub_listener()

    # ------------------------------------------------------------------
    # Redis key helpers
    # ------------------------------------------------------------------

    def _make_key(self, agent_id: str) -> str:
        return f"{self.cfg.key_prefix}:{agent_id}:{uuid.uuid4().hex}"

    def _agent_pattern(self, agent_id: str) -> str:
        return f"{self.cfg.key_prefix}:{agent_id}:*"

    def _all_pattern(self) -> str:
        return f"{self.cfg.key_prefix}:*:*"

    def _lock_key(self, query: str) -> str:
        digest = hashlib.sha256(query.encode()).hexdigest()
        return f"lock:{self.cfg.key_prefix}:{digest}"

    def _stats_hit_key(self, agent_id: str) -> str:
        return f"{self.cfg.key_prefix}:stats:{agent_id}:hits"

    def _stats_miss_key(self, agent_id: str) -> str:
        return f"{self.cfg.key_prefix}:stats:{agent_id}:misses"

    # ------------------------------------------------------------------
    # FAISS index management
    # ------------------------------------------------------------------

    def _rebuild_index(self) -> None:
        """Rebuild local FAISS index from all entries stored in Redis."""
        with self._lock:
            self._flat_index.reset()
            self._id_to_key.clear()
            self._key_to_id.clear()
            self._next_id = 0

            pattern = self._all_pattern()
            cursor = 0
            vectors, ids = [], []

            while True:
                cursor, keys = self._redis.scan(cursor, match=pattern, count=100)
                for raw_key in keys:
                    key = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
                    # Skip stat keys and lock keys
                    if ":stats:" in key or key.startswith("lock:"):
                        continue
                    entry = self._redis.hgetall(raw_key)
                    if not entry or b"embedding" not in entry:
                        continue
                    try:
                        vec = np.frombuffer(entry[b"embedding"], dtype=np.float32).copy()
                        fid = self._next_id
                        self._next_id += 1
                        vectors.append(vec)
                        ids.append(fid)
                        self._id_to_key[fid] = key
                        self._key_to_id[key] = fid
                    except Exception as exc:
                        logger.warning("Skipping corrupt cache entry %s: %s", key, exc)

                if cursor == 0:
                    break

            if vectors:
                mat = np.stack(vectors).astype(np.float32)
                self._index.add_with_ids(mat, np.array(ids, dtype=np.int64))
                logger.info("FAISS index rebuilt with %d vectors", len(vectors))

    def _add_to_faiss(self, key: str, vec: np.ndarray) -> None:
        """Add a single vector to the local FAISS index (call under self._lock)."""
        fid = self._next_id
        self._next_id += 1
        self._index.add_with_ids(
            vec.reshape(1, -1).astype(np.float32),
            np.array([fid], dtype=np.int64),
        )
        self._id_to_key[fid] = key
        self._key_to_id[key] = fid

    def _remove_from_faiss(self, key: str) -> None:
        """Remove a single vector by key — requires a full rebuild."""
        self._rebuild_index()

    # ------------------------------------------------------------------
    # Pub/Sub listener (background thread)
    # ------------------------------------------------------------------

    def _start_pubsub_listener(self) -> None:
        listener_redis = redis.Redis.from_url(
            self.cfg.redis_url, decode_responses=False
        )
        pubsub = listener_redis.pubsub()
        pubsub.subscribe(self.cfg.sync_channel, self.cfg.invalidate_channel)

        def _listen() -> None:
            for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                try:
                    channel = message["channel"]
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode()

                    if channel == self.cfg.sync_channel.encode():
                        self._handle_sync_event(data)
                    elif channel == self.cfg.invalidate_channel.encode():
                        self._handle_invalidate_event(data)
                except Exception as exc:
                    logger.warning("Pub/Sub handler error: %s", exc)

        t = threading.Thread(target=_listen, daemon=True, name="semcache-pubsub")
        t.start()
        logger.info("Pub/Sub listener started")

    def _handle_sync_event(self, key: str) -> None:
        """A peer wrote a new entry; add it to our local FAISS if we don't have it."""
        with self._lock:
            if key in self._key_to_id:
                return  # already present (we were the writer)
            entry = self._redis.hgetall(key)
            if not entry or b"embedding" not in entry:
                return
            vec = np.frombuffer(entry[b"embedding"], dtype=np.float32).copy()
            self._add_to_faiss(key, vec)
            logger.debug("Sync: added %s from peer", key)

    def _handle_invalidate_event(self, payload: str) -> None:
        """A peer invalidated one or more entries; rebuild our index."""
        logger.debug("Invalidate event received: %s", payload)
        with self._lock:
            self._rebuild_index()

    # ------------------------------------------------------------------
    # Public API — get / set / invalidate
    # ------------------------------------------------------------------

    def get(self, query: str, agent_id: str) -> Optional[str]:
        """
        Look up a semantically similar cached response.

        Returns the cached response string, or None on a miss.
        """
        with self._lock:
            if self._index.ntotal == 0:
                self._redis.incr(self._stats_miss_key(agent_id))
                return None

            vec = self._embedder.encode(query).reshape(1, -1)
            distances, ids = self._index.search(vec, k=1)
            cosine_score = float(distances[0][0])
            top_id = int(ids[0][0])

            if top_id == -1 or cosine_score < self.cfg.similarity_threshold:
                self._redis.incr(self._stats_miss_key(agent_id))
                return None

            redis_key = self._id_to_key.get(top_id)
            if not redis_key:
                self._redis.incr(self._stats_miss_key(agent_id))
                return None

            entry = self._redis.hgetall(redis_key)
            if not entry or b"response" not in entry:
                # Stale FAISS entry (TTL expired in Redis); rebuild
                self._rebuild_index()
                self._redis.incr(self._stats_miss_key(agent_id))
                return None

            self._redis.incr(self._stats_hit_key(agent_id))
            logger.debug("Cache HIT (score=%.4f) for agent=%s", cosine_score, agent_id)
            return entry[b"response"].decode()

    def set(
        self,
        query: str,
        response: str,
        agent_id: str,
        ttl: Optional[int] = None,
    ) -> str:
        """
        Store a query/response pair in Redis and update the local FAISS index.
        Publishes a sync event so peer nodes can update their indices.

        Returns the Redis key for the new entry.
        """
        ttl = ttl if ttl is not None else self.cfg.default_ttl
        key = self._make_key(agent_id)
        vec = self._embedder.encode(query)

        entry = {
            "embedding": vec.tobytes(),
            "response": response.encode(),
            "query": query.encode(),
            "agent_id": agent_id.encode(),
            "created_at": str(time.time()).encode(),
        }

        pipe = self._redis.pipeline()
        pipe.hset(key, mapping=entry)
        pipe.expire(key, ttl)
        pipe.publish(self.cfg.sync_channel, key)
        pipe.execute()

        with self._lock:
            self._add_to_faiss(key, vec)

        logger.debug("Cache SET key=%s agent=%s ttl=%ds", key, agent_id, ttl)
        return key

    def invalidate(self, key: str) -> bool:
        """Delete a specific cache entry by its Redis key."""
        deleted = self._redis.delete(key)
        if deleted:
            self._redis.publish(self.cfg.invalidate_channel, key)
            with self._lock:
                self._remove_from_faiss(key)
        return bool(deleted)

    def invalidate_by_agent(self, agent_id: str) -> int:
        """Delete all cache entries for a given agent. Returns count deleted."""
        pattern = self._agent_pattern(agent_id)
        cursor, deleted_count = 0, 0
        keys_to_delete = []

        while True:
            cursor, keys = self._redis.scan(cursor, match=pattern, count=100)
            keys_to_delete.extend(keys)
            if cursor == 0:
                break

        if keys_to_delete:
            deleted_count = self._redis.delete(*keys_to_delete)
            self._redis.publish(
                self.cfg.invalidate_channel,
                json.dumps({"agent_id": agent_id, "count": deleted_count}),
            )
            with self._lock:
                self._rebuild_index()

        return deleted_count

    # ------------------------------------------------------------------
    # Distributed lock + compute
    # ------------------------------------------------------------------

    def cached_or_compute(
        self,
        query: str,
        agent_id: str,
        compute_fn: Callable[[], str],
        ttl: Optional[int] = None,
    ) -> str:
        """
        Return a cached response for *query*, or call *compute_fn* (the LLM)
        exactly once under a distributed Redis lock (preventing cache stampede).

        Args:
            query:       The natural-language query.
            agent_id:   Logical agent identifier for namespacing.
            compute_fn: Zero-argument callable that calls the LLM and returns str.
            ttl:        TTL in seconds for the cached entry (uses config default).

        Returns:
            The response string (from cache or freshly computed).
        """
        # Fast path — cache hit
        cached = self.get(query, agent_id)
        if cached is not None:
            return cached

        lock_key = self._lock_key(query)
        lock_val = uuid.uuid4().hex
        acquired = False

        for attempt in range(self.cfg.max_lock_retries):
            # Try to acquire lock: SET NX EX
            acquired = self._redis.set(
                lock_key, lock_val, nx=True, ex=self.cfg.lock_timeout
            )
            if acquired:
                break

            # Lock held by another node — wait with exponential back-off
            wait = min(0.1 * (2 ** attempt), 2.0)
            time.sleep(wait)

            # Re-check cache (the lock holder may have populated it)
            cached = self.get(query, agent_id)
            if cached is not None:
                return cached

        if not acquired:
            # Gave up waiting — call compute directly (degraded mode)
            logger.warning("Could not acquire lock for query; computing without lock")
            response = compute_fn()
            self.set(query, response, agent_id, ttl)
            return response

        try:
            # Double-check after acquiring lock
            cached = self.get(query, agent_id)
            if cached is not None:
                return cached

            response = compute_fn()
            self.set(query, response, agent_id, ttl)
            return response
        finally:
            # Release lock only if we still own it (Lua script for atomicity)
            release_script = """
            if redis.call('get', KEYS[1]) == ARGV[1] then
                return redis.call('del', KEYS[1])
            else
                return 0
            end
            """
            self._redis.eval(release_script, 1, lock_key, lock_val)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self, agent_id: str) -> dict[str, Any]:
        """Return hit/miss counts and hit rate for the given agent."""
        hits = int(self._redis.get(self._stats_hit_key(agent_id)) or 0)
        misses = int(self._redis.get(self._stats_miss_key(agent_id)) or 0)
        total = hits + misses
        hit_rate = round(hits / total, 4) if total > 0 else 0.0
        return {"hits": hits, "misses": misses, "total": total, "hit_rate": hit_rate}

    # ------------------------------------------------------------------
    # Introspection helpers (used by /health endpoint)
    # ------------------------------------------------------------------

    @property
    def faiss_vector_count(self) -> int:
        with self._lock:
            return self._index.ntotal

    def redis_ping(self) -> bool:
        try:
            return self._redis.ping()
        except Exception:
            return False


# ---------------------------------------------------------------------------
# LangChain integration — SemanticLLMCache
# ---------------------------------------------------------------------------

class SemanticLLMCache(BaseCache):
    """
    Drop-in LangChain BaseCache that uses SemanticCache as its backend.

    Usage::

        from langchain.globals import set_llm_cache
        cache = SemanticLLMCache(SemanticCache())
        set_llm_cache(cache)

    After this, every LangChain LLM call is transparently routed through the
    semantic cache.  The *llm_string* (model identifier) is incorporated into
    the *agent_id* so different models get separate cache namespaces.
    """

    def __init__(self, sem_cache: SemanticCache) -> None:
        self._cache = sem_cache

    # LangChain calls lookup() before the LLM; update() after.

    def lookup(self, prompt: str, llm_string: str) -> Optional[list[Generation]]:
        agent_id = self._agent_id(llm_string)
        raw = self._cache.get(prompt, agent_id)
        if raw is None:
            return None
        try:
            data = json.loads(raw)
            return [Generation(**g) for g in data]
        except Exception:
            return [Generation(text=raw)]

    def update(
        self, prompt: str, llm_string: str, return_val: list[Generation]
    ) -> None:
        agent_id = self._agent_id(llm_string)
        serialised = json.dumps([{"text": g.text} for g in return_val])
        self._cache.set(prompt, serialised, agent_id)

    def clear(self, **kwargs: Any) -> None:
        """Clear all entries in the LangChain-owned namespace."""
        # Invalidate all entries across every agent_id in this namespace
        pattern = f"{self._cache.cfg.key_prefix}:lc-*"
        cursor = 0
        keys_to_delete = []
        while True:
            cursor, keys = self._cache._redis.scan(cursor, match=pattern, count=100)
            keys_to_delete.extend(keys)
            if cursor == 0:
                break
        if keys_to_delete:
            self._cache._redis.delete(*keys_to_delete)
            with self._cache._lock:
                self._cache._rebuild_index()

    @staticmethod
    def _agent_id(llm_string: str) -> str:
        # Use a short hash of the model identifier as the agent namespace
        return "lc-" + hashlib.sha256(llm_string.encode()).hexdigest()[:12]
