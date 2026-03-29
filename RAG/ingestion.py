"""
ingestion.py — Document loading, chunking, and indexing
========================================================
Single responsibility: ingest files into the FAISS vector store.

Knows about document formats and text splitting — not about HTTP routing,
caching, or answer generation.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.vectorstores import FAISS as LangchainFAISS
from langchain_core.documents import Document

from config import settings
from models import get_embeddings
from store import vector_store_manager

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = frozenset({".pdf", ".txt", ".md", ".markdown", ".text"})


class DocumentIngestionService:
    """Load, split, embed, and index documents into the FAISS vector store."""

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def ingest(self, file_paths: list[Path]) -> int:
        """
        Chunk, embed, and add files to the shared FAISS index.

        Returns:
            Number of chunks added (0 if no supported content was found).
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )

        all_chunks: list[Document] = []
        for path in file_paths:
            try:
                docs = self._loader_for(path).load()
                chunks = splitter.split_documents(docs)
                for chunk in chunks:
                    chunk.metadata.setdefault("source", path.name)
                all_chunks.extend(chunks)
                logger.info("Loaded %s — %d chunks", path.name, len(chunks))
            except Exception as exc:
                logger.warning("Failed to load %s: %s", path, exc)

        if not all_chunks:
            return 0

        embeddings = get_embeddings()
        if vector_store_manager.store is None:
            vector_store_manager.store = LangchainFAISS.from_documents(all_chunks, embeddings)
        else:
            vector_store_manager.store.add_documents(all_chunks)

        vector_store_manager.save()
        return len(all_chunks)

    def auto_ingest(self) -> None:
        """Ingest any documents pre-placed in the configured docs volume on startup."""
        docs_dir = Path(settings.docs_path)
        if not docs_dir.exists():
            return
        files = [
            p for p in docs_dir.rglob("*")
            if p.suffix.lower() in _SUPPORTED_EXTENSIONS and p.is_file()
        ]
        if not files:
            return
        logger.info("Auto-ingesting %d file(s) from %s", len(files), settings.docs_path)
        self.ingest(files)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _loader_for(self, path: Path):
        """Select the correct LangChain document loader by file extension."""
        ext = path.suffix.lower()
        if ext == ".pdf":
            return PyPDFLoader(str(path))
        if ext in (".md", ".markdown"):
            return UnstructuredMarkdownLoader(str(path))
        return TextLoader(str(path), encoding="utf-8")


# Module-level singleton — import `ingestion_service` throughout the app.
ingestion_service = DocumentIngestionService()
