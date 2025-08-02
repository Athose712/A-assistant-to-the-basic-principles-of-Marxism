from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_dashscope.embeddings import DashScopeEmbeddings


def load_embeddings(model: str = "text-embedding-v2") -> DashScopeEmbeddings:
    """Initialize and return a DashScopeEmbeddings instance."""
    return DashScopeEmbeddings(model=model)


def load_vectorstore(
    path: str = "database_agent_mayuan",
    embeddings: Optional[DashScopeEmbeddings] = None,
    allow_dangerous_deserialization: bool = True,
):
    """Load a FAISS vectorstore from ``path`` with provided ``embeddings``.

    If *embeddings* is ``None`` a new embedding model will be created with the
    default parameters.
    """
    if embeddings is None:
        embeddings = load_embeddings()

    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=allow_dangerous_deserialization,
    ) 