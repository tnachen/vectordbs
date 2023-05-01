import os
from types import VectorStore

def get_datastore(datastore: str) -> VectorStore:
    assert datastore is not None

    match datastore:
        case "pinecone":
            from providers.pinecone_datastore import PineconeDataStore

            return PineconeDataStore()
        case "weaviate":
            from providers.weaviate_datastore import WeaviateDataStore

            return WeaviateDataStore()
        case "milvus":
            from providers.milvus_datastore import MilvusDataStore

            return MilvusDataStore()
        case "zilliz":
            from providers.zilliz_datastore import ZillizDataStore

            return ZillizDataStore()
        case "redis":
            from providers.redis_datastore import RedisDataStore

            return RedisDataStore.init()
        case "qdrant":
            from providers.qdrant_datastore import QdrantDataStore

            return QdrantDataStore()
        case _:
            raise ValueError(f"Unsupported vector database: {datastore}")
