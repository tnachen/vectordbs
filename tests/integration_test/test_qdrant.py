import uuid
from typing import Optional

import pytest

from vectordbs.types import (
    DocumentChunk,
    DocumentMetadataFilter,
    QueryWithEmbedding,
    DocumentMetadata,
)
from vectordbs.providers.qdrant_datastore import QdrantDataStore

def create_document_chunk(id: Optional[str] = None) -> DocumentChunk:
    return DocumentChunk(
        id=id or str(uuid.uuid4()),
        text="sample text",
        metadata=DocumentMetadata(
            document_id="1",
            source="source",
            source_id="source_id",
            author="author",
        ),
        embedding=[0.1] * 1536,
    )

@pytest.fixture
def qdrant_datastore():
    datastore = QdrantDataStore(recreate_collection=True)
    yield datastore
    datastore.delete(filter=None)


def test_upsert(qdrant_datastore):
    document_chunk = create_document_chunk()
    result = qdrant_datastore.upsert({document_chunk.id: [document_chunk]})
    assert len(result) == 1
    assert result[0] == document_chunk.id


def test_query(qdrant_datastore):
    document_chunk = create_document_chunk()
    qdrant_datastore.upsert({document_chunk.id: [document_chunk]})

    query = QueryWithEmbedding(
        query="test query",
        embedding=[0.1] * 1536,
        filter=DocumentMetadataFilter(document_id=document_chunk.metadata.document_id),
        top_k=5,
    )

    results = qdrant_datastore.query([query])
    assert len(results) == 1
    assert len(results[0].results) >= 1
    assert results[0].results[0].id == document_chunk.id


def test_delete(qdrant_datastore):
    document_chunk = create_document_chunk()
    qdrant_datastore.upsert({document_chunk.id: [document_chunk]})

    deleted = qdrant_datastore.delete(ids=[document_chunk.id])
    assert deleted

    query = QueryWithEmbedding(
        query="test query",
        embedding=[0.1] * 1536,
        filter=DocumentMetadataFilter(document_id=document_chunk.metadata.document_id),
        top_k=5,
    )

    results = qdrant_datastore.query([query])
    assert len(results) == 1
    assert len(results[0].results) == 0
