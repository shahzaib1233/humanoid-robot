"""
Retrieval Result model for the RAG Chatbot application.
Represents the relevant content chunks retrieved from the vector database based on user query.
"""
from sqlalchemy import Column, String, DateTime, Float, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from ..database.connection import Base
from datetime import datetime
import uuid


class RetrievalResult(Base):
    __tablename__ = "retrieval_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_query_id = Column(UUID(as_uuid=True), ForeignKey("user_queries.id"), nullable=False)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("book_content_chunks.id"), nullable=False)
    similarity_score = Column(Float, nullable=False)  # Between 0.0 and 1.0
    rank = Column(Integer, nullable=False)  # Rank position in the retrieval results
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<RetrievalResult(id={self.id}, query_id={self.user_query_id}, chunk_id={self.chunk_id})>"