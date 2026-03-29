"""
models.py — Ollama model singletons (LLM + embeddings)
=======================================================
Single responsibility: create and cache Ollama client instances.

Both the ingestion pipeline (ingestion.py) and the RAG chain (rag_chain.py)
share the same embedding instance to avoid duplicate initialisation overhead.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain_ollama import OllamaEmbeddings, OllamaLLM

from config import settings

logger = logging.getLogger(__name__)

_embeddings: Optional[OllamaEmbeddings] = None
_llm:        Optional[OllamaLLM]        = None


def get_embeddings() -> OllamaEmbeddings:
    """Return the shared OllamaEmbeddings instance (lazy init)."""
    global _embeddings
    if _embeddings is None:
        logger.info("Initialising embeddings model: %s", settings.embedding_model)
        _embeddings = OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )
    return _embeddings


def get_llm() -> OllamaLLM:
    """Return the shared OllamaLLM instance (lazy init)."""
    global _llm
    if _llm is None:
        logger.info("Initialising LLM: %s", settings.ollama_model)
        _llm = OllamaLLM(model=settings.ollama_model, base_url=settings.ollama_base_url)
    return _llm
