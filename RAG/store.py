"""
store.py — FAISS vector store lifecycle management
===================================================
Single responsibility: own the FAISS index — persist it to disk, load it on
startup, and provide a single shared mutable reference for the rest of the app.

No embedding logic, no document loading, no HTTP concerns.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from langchain_community.vectorstores import FAISS as LangchainFAISS

from config import settings

if TYPE_CHECKING:
    from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Lifecycle manager for the in-process FAISS vector store."""

    def __init__(self) -> None:
        self._store: Optional[LangchainFAISS] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def store(self) -> Optional[LangchainFAISS]:
        return self._store

    @store.setter
    def store(self, value: Optional[LangchainFAISS]) -> None:
        self._store = value

    @property
    def vector_count(self) -> int:
        return self._store.index.ntotal if self._store else 0

    @property
    def is_ready(self) -> bool:
        return self._store is not None

    def load_persisted(self, embeddings: OllamaEmbeddings) -> None:
        """Load a saved FAISS index from the configured volume path on startup."""
        index_dir = Path(settings.faiss_index_path)
        if not index_dir.exists() or not any(index_dir.iterdir()):
            return
        try:
            self._store = LangchainFAISS.load_local(
                str(index_dir),
                embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("FAISS index loaded — %d vectors", self.vector_count)
        except Exception as exc:
            logger.warning("Could not load FAISS index: %s — starting fresh", exc)

    def save(self) -> None:
        """Persist the current FAISS index to the configured volume path."""
        if self._store is None:
            return
        Path(settings.faiss_index_path).mkdir(parents=True, exist_ok=True)
        self._store.save_local(settings.faiss_index_path)
        logger.info("FAISS index saved — %d vectors", self.vector_count)

    def clear(self) -> None:
        """Wipe the in-memory store and delete the persisted index from disk."""
        self._store = None
        index_dir = Path(settings.faiss_index_path)
        if index_dir.exists():
            shutil.rmtree(index_dir)
        logger.info("FAISS index cleared")


# Module-level singleton — import `vector_store_manager` throughout the app.
vector_store_manager = VectorStoreManager()
