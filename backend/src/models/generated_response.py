"""
Generated Response model for the RAG Chatbot application.
Represents the LLM-generated answer to a user query.
"""
from sqlalchemy import Column, Text, DateTime, Float, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.mutable import MutableDict
from ..database.connection import Base
from datetime import datetime
import uuid


class GeneratedResponse(Base):
    __tablename__ = "generated_responses"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_query_id = Column(UUID(as_uuid=True), ForeignKey("user_queries.id"), nullable=False, unique=True)
    response_text = Column(Text, nullable=False)
    citations = Column(MutableDict.as_mutable(JSONB), default=list)  # List of citation objects
    confidence_indicator = Column(Float, nullable=True)  # Between 0.0 and 1.0
    timestamp = Column(DateTime, default=datetime.utcnow)
    token_usage = Column(MutableDict.as_mutable(JSONB), default={})  # Token usage information

    def __repr__(self):
        return f"<GeneratedResponse(id={self.id}, query_id={self.user_query_id})>"