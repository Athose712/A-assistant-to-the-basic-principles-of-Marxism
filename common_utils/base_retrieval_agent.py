"""
common_utils.base_retrieval_agent
================================

A lightweight reusable base class for retrieval-augmented agents that need to

* initialise embeddings + LLM (DashScope)
* load a FAISS vector store
* provide a helper to retrieve document snippets

It intentionally keeps no LangGraph workflow so that child agents can build
custom logic easily (e.g. Q&A, summarisation). This mirrors the functionality
that already exists in *base_agent.py* and *base_kg_agent.py* but without the
question-generation specific graph.
"""

from __future__ import annotations

import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_dashscope.embeddings import DashScopeEmbeddings

from .llm_wrapper import CustomChatDashScope

__all__ = ["BaseRetrievalAgent"]


class BaseRetrievalAgent:
    """Reusable mix-in providing retrieval utilities for subject-specific agents."""

    def __init__(
        self,
        *,
        subject_name: str,
        vectorstore_path: str,
        llm_model: str = "qwen-max",
        embedding_model: str = "text-embedding-v2",
        temperature: float = 0.3,
    ) -> None:
        self.subject_name = subject_name
        self.vectorstore_path = vectorstore_path

        if "DASHSCOPE_API_KEY" not in os.environ:
            raise EnvironmentError("Please set the DASHSCOPE_API_KEY environment variable.")

        # Model init -----------------------------------------------------------
        try:
            self.embeddings = DashScopeEmbeddings(model=embedding_model)
            self.llm = CustomChatDashScope(model=llm_model, temperature=temperature)
            print(f"[{self.subject_name}] BaseRetrievalAgent – models initialised.")
        except Exception as exc:
            raise RuntimeError(f"Model initialisation failed: {exc}") from exc

        # Vector store ---------------------------------------------------------
        self.vectorstore = self._load_vectorstore()
        # Generation & retrieval controls (can be overridden per request)
        self.generation_kwargs: dict = {}
        self.retrieval_k: int = 5

    def set_generation_params(self, *, max_tokens: int | None = None, timeout: int | None = None, retrieval_k: int | None = None) -> None:
        """Update generation and retrieval controls for subsequent calls."""
        if max_tokens is not None:
            self.generation_kwargs["max_tokens"] = max_tokens
        if timeout is not None:
            self.generation_kwargs["timeout"] = timeout
        if retrieval_k is not None and retrieval_k > 0:
            self.retrieval_k = retrieval_k

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_vectorstore(self) -> Optional[FAISS]:
        """Attempt to load the FAISS vector store at *vectorstore_path*."""
        try:
            print(f"[{self.subject_name}] Loading vector store from '{self.vectorstore_path}' …")
            return FAISS.load_local(
                self.vectorstore_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as exc:
            print(f"[{self.subject_name}] ⚠️  Vector store load failed: {exc}. Proceeding without retrieval.")
            return None

    def _retrieve_docs(self, query: str, k: int = 5) -> List[str]:
        """Return up to *k* unique document snippets relevant to *query*."""
        if self.vectorstore is None:
            return []
        docs = self.vectorstore.similarity_search(query, k=self.retrieval_k or k)
        seen = set()
        snippets: List[str] = []
        for doc in docs:
            content = doc.page_content.strip()
            if content and content not in seen:
                snippets.append(content)
                seen.add(content)
        return snippets
