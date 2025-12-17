"""
Chat endpoint for the RAG Chatbot application.
Handles user queries and returns RAG-powered responses.
"""
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from src.services.rag_service import RAGService, RAGQuery, RAGResult
from src.database.connection import get_db
from sqlalchemy.orm import Session
from slowapi import Limiter
from slowapi.util import get_remote_address
import uuid
import json

# Create limiter for this router
limiter = Limiter(key_func=get_remote_address)


router = APIRouter()

# Initialize RAG service
rag_service = RAGService()


class ChatRequest(BaseModel):
    query: str
    selected_text: Optional[str] = None
    mode: str = "normal"  # "normal" or "selection-only"
    session_id: Optional[str] = None
    stream: bool = False  # Whether to stream the response
    model: Optional[str] = None  # Model to use: "qwen", "gemini", "openrouter", or None for default


class Citation(BaseModel):
    source: str
    content: str
    similarity_score: float


class ChatResponse(BaseModel):
    response: str
    citations: List[Citation]
    session_id: str
    tokens_used: int


def format_stream_response(response: str, citations: List[Dict], session_id: str, tokens_used: int):
    """Format a response chunk for streaming."""
    chunk = {
        "response": response,
        "citations": citations,
        "session_id": session_id,
        "tokens_used": tokens_used
    }
    return f"data: {json.dumps(chunk)}\n\n"


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")  # 10 requests per minute per IP
async def chat_endpoint(request: Request, chat_request: ChatRequest):
    """
    Generate a RAG-powered response to a user query.

    Args:
        request: Chat request containing the user's query and parameters

    Returns:
        ChatResponse with the generated response and citations
    """
    try:
        # Generate a session ID if not provided
        session_id = chat_request.session_id or str(uuid.uuid4())

        # Create RAG query object
        rag_query = RAGQuery(
            query=chat_request.query,
            selected_text=chat_request.selected_text,
            mode=chat_request.mode,
            session_id=session_id,
            model=chat_request.model
        )

        # If streaming is requested, return a streaming response
        if chat_request.stream:
            async def generate_stream():
                # Process the query using RAG service streaming
                all_citations = []

                for chunk_data in rag_service.query_stream(rag_query):
                    # Format citations if they exist in this chunk (they should only exist in the first chunk)
                    if chunk_data["citations"]:
                        all_citations = [
                            Citation(
                                source=citation["source"],
                                content=citation["content"],
                                similarity_score=citation["similarity_score"]
                            )
                            for citation in chunk_data["citations"]
                        ]

                    # Format and yield the stream response
                    yield format_stream_response(
                        chunk_data["response"],
                        [c.dict() for c in all_citations] if chunk_data["citations"] else [],
                        session_id,
                        chunk_data["tokens_used"]
                    )

            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        else:
            # Process the query using RAG service (non-streaming)
            result: RAGResult = rag_service.query(rag_query)

            # Format citations for response
            formatted_citations = [
                Citation(
                    source=citation["source"],
                    content=citation["content"],
                    similarity_score=citation["similarity_score"]
                )
                for citation in result.citations
            ]

            return ChatResponse(
                response=result.response,
                citations=formatted_citations,
                session_id=session_id,
                tokens_used=result.tokens_used
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")