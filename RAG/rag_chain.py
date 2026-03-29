"""
rag_chain.py — RAG retrieval and generation chain
==================================================
Single responsibility: given a question, retrieve relevant document chunks
from FAISS and produce a grounded answer via the Ollama LLM.

No cache logic, no HTTP routing, no document loading.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from models import get_llm
from store import vector_store_manager

logger = logging.getLogger(__name__)

_RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using ONLY the context provided below.
If the answer is not contained in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
)


class RAGChain:
    """Retrieval-Augmented Generation: FAISS retriever → prompt → Ollama LLM."""

    def run(self, question: str, top_k: int) -> dict:
        """
        Retrieve the top-k most relevant chunks and generate a grounded answer.

        Returns:
            dict with keys: answer (str), sources (list[str]), chunks_retrieved (int)

        Raises:
            ValueError: if no documents have been ingested yet.
        """
        store = vector_store_manager.store
        if store is None:
            raise ValueError("No documents ingested")

        retriever = store.as_retriever(search_kwargs={"k": top_k})
        chain = (
            {"context": retriever | self._format_docs, "question": RunnablePassthrough()}
            | _RAG_PROMPT
            | get_llm()
            | StrOutputParser()
        )

        retrieved_docs: list[Document] = retriever.invoke(question)
        answer: str = chain.invoke(question)
        sources = sorted({d.metadata.get("source", "unknown") for d in retrieved_docs})

        return {
            "answer": answer,
            "sources": sources,
            "chunks_retrieved": len(retrieved_docs),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_docs(self, docs: list[Document]) -> str:
        return "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in docs
        )


# Module-level singleton — import `rag_chain` throughout the app.
rag_chain = RAGChain()
