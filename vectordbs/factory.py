import os
from vectordbs.types import VectorStore

def get_datastore(datastore: str) -> VectorStore:
    assert datastore is not None

    match datastore:
        case "pinecone":
            from vectordbs.providers.pinecone_datastore import PineconeDataStore

            return PineconeDataStore()
        case "weaviate":
            from vectordbs.providers.weaviate_datastore import WeaviateDataStore

            return WeaviateDataStore()
        case "milvus":
            from vectordbs.providers.milvus_datastore import MilvusDataStore

            return MilvusDataStore()
        case "zilliz":
            from vectordbs.providers.zilliz_datastore import ZillizDataStore

            return ZillizDataStore()
        case "redis":
            from vectordbs.providers.redis_datastore import RedisDataStore

            return RedisDataStore.init()
        case "qdrant":
            from vectordbs.providers.qdrant_datastore import QdrantDataStore

            return QdrantDataStore()
        case _:
            raise ValueError(f"Unsupported vector database: {datastore}")
