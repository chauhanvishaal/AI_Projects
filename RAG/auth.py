"""
auth.py — API key authentication dependency
============================================
Single responsibility: verify the X-API-Key header for all protected routes.
Uses secrets.compare_digest to prevent timing-based side-channel attacks.
"""

from __future__ import annotations

import logging
import secrets

from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader

from config import settings

logger = logging.getLogger(__name__)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str = Security(_api_key_header)) -> None:
    """FastAPI dependency — raises 403 if the API key is invalid or absent."""
    if not settings.api_key:
        logger.warning("API_KEY is not set; running without authentication")
        return
    if not api_key or not secrets.compare_digest(api_key, settings.api_key):
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
