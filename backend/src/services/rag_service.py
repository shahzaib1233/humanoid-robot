"""
RAG (Retrieval-Augmented Generation) service for the chatbot application.
Coordinates embedding, vector search, and LLM generation.
"""
import logging
from typing import List, Dict, Any, Optional
from ..services.embedding_service import EmbeddingService
from ..services.vector_db_service import VectorDBService
from ..services.multi_llm_service import MultiLLMService
from pydantic import BaseModel

# Set up logging
logger = logging.getLogger(__name__)


class RAGQuery(BaseModel):
    query: str
    selected_text: Optional[str] = None
    mode: str = "normal"  # "normal" or "selection-only"
    session_id: Optional[str] = None
    model: Optional[str] = None  # "qwen", "gemini", "openrouter", or None for default


class RAGResult(BaseModel):
    response: str
    citations: List[Dict[str, Any]]
    tokens_used: int


class RAGService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_db_service = VectorDBService()
        self.llm_service = MultiLLMService()

    def query(self, rag_query: RAGQuery) -> RAGResult:
        """
        Process a RAG query and return a response with citations.

        Args:
            rag_query: Query object containing the user's question and parameters

        Returns:
            RAGResult object with response and citations
        """
        logger.info(f"Processing RAG query: {rag_query.query[:50]}...")

        # If in selection-only mode, use only the selected text
        if rag_query.mode == "selection-only" and rag_query.selected_text:
            context = rag_query.selected_text
            logger.info("Using selection-only mode")

            # If the selected text is insufficient, return a specific message
            if not context.strip() or len(context.strip()) < 10:
                logger.info("Insufficient selected text, returning appropriate message")
                return RAGResult(
                    response="I don't have enough info in the selected text.",
                    citations=[],
                    tokens_used=0
                )
        else:
            logger.info("Using normal mode, retrieving context from vector database")
            # In normal mode, retrieve relevant context from the vector database
            query_embedding = self.embedding_service.create_embedding(rag_query.query)
            similar_chunks = self.vector_db_service.search_similar(
                query_vector=query_embedding,
                limit=5
            )

            # Build context from retrieved chunks
            context_parts = []
            citations = []

            for chunk in similar_chunks:
                content = chunk["payload"].get("content_text", "")
                source = chunk["payload"].get("source_document_ref", "")
                similarity_score = chunk["similarity_score"]

                context_parts.append(content)
                citations.append({
                    "source": source,
                    "content": content[:200] + "..." if len(content) > 200 else content,  # Truncate for display
                    "similarity_score": similarity_score
                })

            context = "\n\n".join(context_parts)
            logger.info(f"Retrieved {len(similar_chunks)} relevant chunks for query")

        # Generate response using the LLM
        llm_response = self.llm_service.generate_response(
            prompt=rag_query.query,
            context=context,
            model=rag_query.model
        )

        logger.info(f"Generated response with {llm_response.tokens_used} tokens used")

        return RAGResult(
            response=llm_response.content,
            citations=citations,
            tokens_used=llm_response.tokens_used
        )

    def query_stream(self, rag_query: RAGQuery):
        """
        Process a RAG query and return a streaming response with citations.
        Yields response chunks as they are generated.

        Args:
            rag_query: Query object containing the user's question and parameters
        """
        import logging
        logger = logging.getLogger(__name__)

        logger.info(f"Processing streaming RAG query: {rag_query.query[:50]}...")

        # If in selection-only mode, use only the selected text
        if rag_query.mode == "selection-only" and rag_query.selected_text:
            context = rag_query.selected_text
            logger.info("Using selection-only mode")

            # If the selected text is insufficient, return a specific message
            if not context.strip() or len(context.strip()) < 10:
                logger.info("Insufficient selected text, returning appropriate message")
                yield {
                    "response": "I don't have enough info in the selected text.",
                    "citations": [],
                    "tokens_used": 0
                }
        else:
            logger.info("Using normal mode, retrieving context from vector database")
            # In normal mode, retrieve relevant context from the vector database
            query_embedding = self.embedding_service.create_embedding(rag_query.query)
            similar_chunks = self.vector_db_service.search_similar(
                query_vector=query_embedding,
                limit=5
            )

            # Build context from retrieved chunks
            context_parts = []
            citations = []

            for chunk in similar_chunks:
                content = chunk["payload"].get("content_text", "")
                source = chunk["payload"].get("source_document_ref", "")
                similarity_score = chunk["similarity_score"]

                context_parts.append(content)
                citations.append({
                    "source": source,
                    "content": content[:200] + "..." if len(content) > 200 else content,  # Truncate for display
                    "similarity_score": similarity_score
                })

            context = "\n\n".join(context_parts)
            logger.info(f"Retrieved {len(similar_chunks)} relevant chunks for query")

        # Stream response using the LLM
        token_count = 0
        full_response = ""

        # Send citations first in the stream
        yield {
            "response": "",
            "citations": citations,
            "tokens_used": 0
        }

        # Stream the response content
        for token in self.llm_service.generate_response_stream(
            prompt=rag_query.query,
            context=context,
            model=rag_query.model
        ):
            full_response += token
            token_count += 1  # This is a rough estimate; actual token counting would require more sophisticated tracking

            yield {
                "response": token,
                "citations": [],  # Citations already sent in the first chunk
                "tokens_used": token_count
            }

    def index_content(self, content_text: str, source_ref: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Index content by creating embeddings and storing them in the vector database.

        Args:
            content_text: The text content to index
            source_ref: Reference to the source document
            metadata: Additional metadata to store with the content

        Returns:
            True if indexing was successful, False otherwise
        """
        logger.info(f"Indexing content from {source_ref}, length: {len(content_text)} characters")

        if metadata is None:
            metadata = {}

        # Create embedding for the content
        embedding = self.embedding_service.create_embedding(content_text)

        # Prepare payload for vector database
        payload = {
            "content_text": content_text,
            "source_document_ref": source_ref,
            "metadata": metadata
        }

        # Generate a unique ID for this chunk (in a real implementation, you'd use a proper ID generation)
        import uuid
        chunk_id = str(uuid.uuid4())

        # Store in vector database
        success = self.vector_db_service.store_embedding(
            record_id=chunk_id,
            vector=embedding,
            payload=payload
        )

        if success:
            logger.info(f"Successfully indexed content chunk {chunk_id} from {source_ref}")
        else:
            logger.error(f"Failed to index content chunk from {source_ref}")

        return success