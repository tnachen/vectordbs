from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from enum import Enum
from abc import ABC, abstractmethod
from pydantic import BaseModel

@dataclass
class DocumentChunk:
    document_id: str
    text: str
    vector: List[float]

@dataclass
class DocumentMetadataFilter:
    field_name: str
    gte: Optional[int] = None
    lte: Optional[int] = None

@dataclass
class DocumentChunkWithScore(DocumentChunk):
    score: float

@dataclass
class QueryResult:
    data: Optional[List[Any]] = None
    similarities: Optional[List[float]] = None
    ids: Optional[List[str]] = None

@dataclass
class QueryWithEmbedding:
    text: str
    vector: List[float]

@dataclass
class VectorStoreData:
    id: str
    data: dict
    embedding: List[float]

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
    """Abstract vector store class."""
    
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
    ) -> QueryResult:
        """Query vector store."""
        ...
