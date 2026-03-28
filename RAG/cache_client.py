"""
cache_client.py — HTTP gateway to the Semantic Cache service
=============================================================
Single responsibility: communicate with semcache-api over HTTP.

This module knows nothing about RAG, documents, FAISS, or LLMs.
All cache concerns (Redis, vector search, TTL, distributed locks)
are handled entirely within the semcache-api service.

If the cache service is unavailable the client degrades gracefully —
every method swallows connectivity errors so the RAG pipeline
continues to function without caching.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

SEMCACHE_BASE_URL = os.environ.get("SEMCACHE_BASE_URL", "http://semcache-api:8000")
SEMCACHE_API_KEY  = os.environ.get("SEMCACHE_API_KEY", "")
SEMCACHE_AGENT_ID = os.environ.get("SEMCACHE_AGENT_ID", "rag-workflow")
SEMCACHE_TTL      = int(os.environ.get("SEMCACHE_TTL", "3600"))
SEMCACHE_TIMEOUT  = float(os.environ.get("SEMCACHE_TIMEOUT", "5.0"))


class SemanticCacheClient:
    """
    Thin HTTP client for the semcache-api service.

    Public interface (two methods only):
      lookup(question) -> str | None   — cache hit returns answer, miss returns None
      store(question, answer)          — populate cache after a RAG miss
    """

    def __init__(
        self,
        base_url: str = SEMCACHE_BASE_URL,
        api_key: str = SEMCACHE_API_KEY,
        agent_id: str = SEMCACHE_AGENT_ID,
        ttl: int = SEMCACHE_TTL,
        timeout: float = SEMCACHE_TIMEOUT,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._agent_id = agent_id
        self._ttl = ttl
        self._timeout = timeout
        self._headers = {"X-API-Key": api_key, "Content-Type": "application/json"} if api_key else {"Content-Type": "application/json"}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, question: str) -> Optional[str]:
        """
        Check the semantic cache for a previously answered similar question.

        Returns the cached answer string on a hit, or None on a miss.
        Connectivity failures return None so RAG continues uninterrupted.
        """
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(
                    f"{self._base_url}/query",
                    headers=self._headers,
                    json={"query": question, "agent_id": self._agent_id},
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("cache_hit"):
                    logger.debug("Semantic cache HIT (agent=%s)", self._agent_id)
                    return data["response"]
                return None
        except httpx.RequestError as exc:
            logger.warning("Semantic cache unreachable: %s — proceeding without cache", exc)
            return None
        except httpx.HTTPStatusError as exc:
            logger.warning("Semantic cache HTTP %s — proceeding without cache", exc.response.status_code)
            return None

    def store(self, question: str, answer: str) -> None:
        """
        Store a new question/answer pair in the cache (PUT /cache).

        Failures are logged and swallowed — a cache write must never
        surface as an error to the RAG caller.
        """
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.put(
                    f"{self._base_url}/cache",
                    headers=self._headers,
                    json={
                        "query": question,
                        "response": answer,
                        "agent_id": self._agent_id,
                        "ttl": self._ttl,
                    },
                )
                resp.raise_for_status()
                logger.debug("Semantic cache STORE ok (agent=%s)", self._agent_id)
        except httpx.RequestError as exc:
            logger.warning("Semantic cache store unreachable: %s (non-fatal)", exc)
        except httpx.HTTPStatusError as exc:
            logger.warning("Semantic cache store HTTP %s (non-fatal)", exc.response.status_code)

    def is_available(self) -> bool:
        """Probe the cache service /health endpoint (unauthenticated)."""
        try:
            with httpx.Client(timeout=3.0) as client:
                return client.get(f"{self._base_url}/health").status_code == 200
        except Exception:
            return False
