"""
config.py — Application settings
=================================
Single responsibility: read configuration from environment variables and
expose a typed, immutable Settings dataclass used throughout the application.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    ollama_base_url:  str = field(default_factory=lambda: os.environ.get("OLLAMA_BASE_URL",     "http://localhost:11434"))
    ollama_model:     str = field(default_factory=lambda: os.environ.get("OLLAMA_MODEL",        "llama3"))
    embedding_model:  str = field(default_factory=lambda: os.environ.get("EMBEDDING_MODEL",     "nomic-embed-text"))
    faiss_index_path: str = field(default_factory=lambda: os.environ.get("FAISS_INDEX_PATH",    "/data/faiss_index"))
    docs_path:        str = field(default_factory=lambda: os.environ.get("DOCS_PATH",           "/data/documents"))
    chunk_size:       int = field(default_factory=lambda: int(os.environ.get("CHUNK_SIZE",      "512")))
    chunk_overlap:    int = field(default_factory=lambda: int(os.environ.get("CHUNK_OVERLAP",   "64")))
    retrieval_top_k:  int = field(default_factory=lambda: int(os.environ.get("RETRIEVAL_TOP_K", "4")))
    api_key:          str = field(default_factory=lambda: os.environ.get("API_KEY",             ""))


# Module-level singleton — import `settings` throughout the application.
settings = Settings()
