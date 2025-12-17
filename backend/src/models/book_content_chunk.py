"""
Book Content Chunk model for the RAG Chatbot application.
Represents a segment of book content that has been processed and embedded for retrieval.
"""
from sqlalchemy import Column, String, Text, DateTime, Float, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.mutable import MutableDict
from ..database.connection import Base
from datetime import datetime
import uuid


class BookContentChunk(Base):
    __tablename__ = "book_content_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_text = Column(Text, nullable=False)
    embedding_vector = Column(String, nullable=False)  # In practice, you'd use a proper vector column
    source_document_ref = Column(String, nullable=False, index=True)
    metadata = Column(MutableDict.as_mutable(JSONB), default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<BookContentChunk(id={self.id}, source='{self.source_document_ref}')>"