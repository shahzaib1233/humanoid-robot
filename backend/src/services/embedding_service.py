"""
Embedding service for the RAG Chatbot application.
Handles text embedding using OpenAI-compatible API.
"""
import os
import requests
import logging
from typing import List
from pydantic import BaseModel

# Set up logging
logger = logging.getLogger(__name__)


class EmbeddingRequest(BaseModel):
    input: str
    model: str = "text-embedding-ada-002"


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str


class EmbeddingService:
    def __init__(self):
        self.api_key = os.getenv("QWEN_EMBEDDING_API_KEY", "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")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1") + "/embeddings"

    def create_embedding(self, text: str) -> List[float]:
        """
        Create embedding for the given text using OpenAI-compatible API.

        Args:
            text: Input text to embed

        Returns:
            List of embedding values
        """
        logger.info(f"Creating embedding for text of length {len(text)}")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "input": text,
            "model": "text-embedding-ada-002"
        }

        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

            # Extract the embedding from the response
            embedding = data['data'][0]['embedding']
            logger.info(f"Successfully created embedding with {len(embedding)} dimensions")
            return embedding
        except Exception as e:
            import traceback
            logger.error(f"Error calling embedding API: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Return a mock embedding as fallback to allow the flow to continue and see LLM errors
            return [0.0] * 1536  # Standard embedding size

    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a batch of texts.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "input": texts,
                "model": "text-embedding-ada-002"
            }

            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

            embeddings = [item['embedding'] for item in data['data']]
            return embeddings
        except Exception as e:
            logger.error(f"Error calling embedding API for batch: {e}")
            # Fallback to individual processing
            return [self.create_embedding(text) for text in texts]