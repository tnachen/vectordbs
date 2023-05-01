"""CREDIT: Originated from GPT index code"""
"""Vector store index types."""


from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from enum import Enum
from abc import ABC, abstractmethod

@dataclass
class VectorStoreData:
    id: str
    data: dict
    embedding: List[float]


@dataclass
class VectorStoreQueryResult:
    """Vector store query result."""

    data: Optional[List[Any]] = None
    similarities: Optional[List[float]] = None
    ids: Optional[List[str]] = None


class VectorStoreQueryMode(str, Enum):
    """Vector store query mode."""

    DEFAULT = "default"
    SPARSE = "sparse"
    HYBRID = "hybrid"


@dataclass
class VectorStoreQuery:
    """Vector store query."""

    # dense embedding
    query_embedding: Optional[List[float]] = None
    similarity_top_k: int = 1
    ids: Optional[List[str]] = None
    query_str: Optional[str] = None

    # NOTE: current mode
    mode: VectorStoreQueryMode = VectorStoreQueryMode.DEFAULT

    # NOTE: only for hybrid search (0 for bm25, 1 for vector search)
    alpha: Optional[float] = None


class VectorStore(ABC):
    """Abstract vector store protocol."""

    @abstractmethod
    def add(
        self,
        datas: List[VectorStoreData],
    ) -> List[str]:
        """Add embedding results to vector store."""
        ...

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete doc."""
        ...

    @abstractmethod
    def query(
        self,
        query: VectorStoreQuery,
    ) -> VectorStoreQueryResult:
        """Query vector store."""
        ...
