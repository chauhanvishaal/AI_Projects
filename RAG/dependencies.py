"""
dependencies.py — Shared FastAPI dependencies
=============================================
Single responsibility: provide injectable application-level dependencies
that are shared across multiple routers.

Keeps router files free of instantiation concerns and makes mocking trivial
in tests (override the dependency via app.dependency_overrides).
"""

from __future__ import annotations

from typing import Optional

from cache_client import SemanticCacheClient

_cache_client: Optional[SemanticCacheClient] = None


def get_cache_client() -> SemanticCacheClient:
    """Return the shared SemanticCacheClient instance (lazy init)."""
    global _cache_client
    if _cache_client is None:
        _cache_client = SemanticCacheClient()
    return _cache_client
