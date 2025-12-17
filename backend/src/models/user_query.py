"""
User Query model for the RAG Chatbot application.
Represents a question or request from a user.
"""
from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.mutable import MutableDict
from ..database.connection import Base
from datetime import datetime
import uuid


class UserQuery(Base):
    __tablename__ = "user_queries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    chat_session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id"), nullable=False)
    query_text = Column(Text, nullable=False)
    selected_text = Column(Text, nullable=True)  # Text selected by user on the page
    mode = Column(String, nullable=False, default="normal")  # 'normal' or 'selection-only'
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    metadata = Column(MutableDict.as_mutable(JSONB), default={})

    def __repr__(self):
        return f"<UserQuery(id={self.id}, session_id={self.chat_session_id})>"