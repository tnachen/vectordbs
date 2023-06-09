import os
from typing import Any, Dict, List, Optional
import pinecone
import asyncio
from pydantic import BaseSettings, Field

from vectordbs.types import VectorStore, VectorStoreData, VectorStoreQuery, VectorStoreQueryResult

# Set the batch size for upserting vectors to Pinecone
UPSERT_BATCH_SIZE = 100

class PineconeOptions(BaseSettings):
    api_key: str = Field(..., env="PINECONE_API_KEY")
    environment: str = Field(..., env="PINECONE_ENVIRONMENT")
    index: str = Field(..., env="PINECONE_INDEX")

class PineconeDataStore(VectorStore):
    def __init__(self, options: PineconeOptions):
        
        # Initialize Pinecone with the API key and environment
        # NOTE: Do we need a singleton to make sure we only init once?
        pinecone.init(api_key=options.api_key, environment=options.environment)

        # Will raise if index doesn't exist
        self.index = pinecone.Index(options.index)        

    async def _upsert(self, datas: List[VectorStoreData]) -> List[str]:
        """
        Takes in a dict from document id to list of document chunks and inserts them into the index.
        Return a list of document ids.
        """
        # Initialize a list of ids to return
        doc_ids: List[str] = []
        # Initialize a list of vectors to upsert
        vectors = []
        # Loop through the dict items
        for data in datas:
            # Append the id to the ids list
            doc_ids.append(data.id)
            vector = (chunk.id, chunk.embedding, chunk.metadata)
            vectors.append(vector)                

        # Split the vectors list into batches of the specified size
        batches = [
            vectors[i : i + UPSERT_BATCH_SIZE]
            for i in range(0, len(vectors), UPSERT_BATCH_SIZE)
        ]
        # Upsert each batch to Pinecone
        for batch in batches:
            try:
                print(f"Upserting batch of size {len(batch)}")
                self.index.upsert(vectors=batch)
                print(f"Upserted batch successfully")
            except Exception as e:
                print(f"Error upserting batch: {e}")
                raise e

        return doc_ids

    async def _query(
        self,
        queries: List[VectorStoreQuery],
    ) -> List[VectorStoreQueryResult()]:
        """
        Takes in a list of queries with embeddings and filters and returns a list of query results with matching document chunks and scores.
        """

        # Define a helper coroutine that performs a single query and returns a QueryResult
        async def _single_query(query: VectorStoreQuery) -> VectorStoreQueryResult():
            #print(f"Query: {query.query}")

            # Convert the metadata filter object to a dict with pinecone filter expressions
            pinecone_filter = self._get_pinecone_filter(query.filter)

            try:
                # Query the index with the query embedding, filter, and top_k
                query_response = self.index.query(
                    # namespace=namespace,
                    top_k=query.top_k,
                    vector=query.embedding,
                    filter=pinecone_filter,
                    include_metadata=True,
                )
            except Exception as e:
                print(f"Error querying index: {e}")
                raise e

            query_results: List[DocumentChunkWithScore] = []
            for result in query_response.matches:
                score = result.score
                metadata = result.metadata
                # Remove document id and text from metadata and store it in a new variable
                metadata_without_text = (
                    {key: value for key, value in metadata.items() if key != "text"}
                    if metadata
                    else None
                )

                # If the source is not a valid Source in the Source enum, set it to None
                if (
                    metadata_without_text
                    and "source" in metadata_without_text
                    and metadata_without_text["source"] not in Source.__members__
                ):
                    metadata_without_text["source"] = None

                # Create a document chunk with score object with the result data
                result = DocumentChunkWithScore(
                    id=result.id,
                    score=score,
                    text=metadata["text"] if metadata and "text" in metadata else None,
                    metadata=metadata_without_text,
                )
                query_results.append(result)
            return QueryResult(query=query.query, results=query_results)

        # Use asyncio.gather to run multiple _single_query coroutines concurrently and collect their results
        results: List[QueryResult] = await asyncio.gather(
            *[_single_query(query) for query in queries]
        )

        return results

    async def delete(
        self, ids: List[str]) -> bool:
        """
        Removes vectors by ids.
        """
        try:
            print(f"Deleting vectors with ids {ids}")
            self.index.delete(ids=ids)  # type: ignore
            print(f"Deleted vectors with ids successfully")
        except Exception as e:
            print(f"Error deleting vectors with ids: {e}")
            raise e

        return True