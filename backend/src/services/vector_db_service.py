"""
Vector database service for the RAG Chatbot application.
Handles interactions with Qdrant Cloud for vector similarity search.
"""
import os
import logging
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from pydantic import BaseModel

# Set up logging
logger = logging.getLogger(__name__)


class VectorRecord(BaseModel):
    id: str
    vector: List[float]
    payload: Dict[str, Any]


class VectorDBService:
    def __init__(self):
        # Use the provided Qdrant settings (defaulting to localhost)
        self.api_key = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.6Y_aEy4ceXSYjm9nWzoAKUl7vwDT8btL7AhSEDJYLXs")
        self.url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.port = int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "book_content_chunks")

        # Initialize Qdrant client
        # For localhost, we don't need API key, for cloud we do
        if "localhost" in self.url or "127.0.0.1" in self.url:
            self.client = QdrantClient(
                host="localhost",
                port=self.port
            )
        else:
            # For cloud instances, use the URL and API key
            # Updated to use correct format for Qdrant Cloud
            self.client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                prefer_grpc=False,
                https=True  # Qdrant Cloud requires HTTPS
            )

        # Create collection if it doesn't exist
        try:
            self._ensure_collection_exists()
        except Exception as e:
            print(f"Warning: Could not connect to Qdrant: {e}")
            print("Make sure Qdrant server is running before using the service")
            # Don't fail initialization if Qdrant is not available during startup

    def _ensure_collection_exists(self):
        """Ensure the collection exists with the proper configuration."""
        try:
            # Check if collection exists
            self.client.get_collection(self.collection_name)
        except:
            # Create collection if it doesn't exist
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1536,  # Assuming 1536 dimensions like OpenAI embeddings
                    distance=models.Distance.COSINE
                )
            )

    def store_embedding(self, record_id: str, vector: List[float], payload: Dict[str, Any]) -> bool:
        """
        Store an embedding in the vector database.

        Args:
            record_id: Unique identifier for the record
            vector: Embedding vector
            payload: Additional metadata to store with the embedding

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=record_id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            return True
        except Exception as e:
            print(f"Error storing embedding: {e}")
            return False

    def search_similar(self, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the database.

        Args:
            query_vector: The vector to search for similar matches
            limit: Maximum number of results to return

        Returns:
            List of similar records with their payload data
        """
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )

            # Format results to include payload and similarity score
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.id,
                    "payload": result.payload,
                    "similarity_score": result.score
                })

            return formatted_results
        except Exception as e:
            print(f"Error searching for similar vectors: {e}")
            # Provide more specific error information for debugging
            import traceback
            traceback.print_exc()
            return []

    def batch_store_embeddings(self, records: List[VectorRecord]) -> bool:
        """
        Store multiple embeddings in the vector database.

        Args:
            records: List of VectorRecord objects to store

        Returns:
            True if successful, False otherwise
        """
        try:
            points = [
                models.PointStruct(
                    id=record.id,
                    vector=record.vector,
                    payload=record.payload
                ) for record in records
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            return True
        except Exception as e:
            print(f"Error storing batch embeddings: {e}")
            return False

    def delete_record(self, record_id: str) -> bool:
        """
        Delete a record from the vector database.

        Args:
            record_id: ID of the record to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[record_id]
                )
            )
            return True
        except Exception as e:
            print(f"Error deleting record: {e}")
            return False