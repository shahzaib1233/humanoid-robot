"""
Test script to verify the streaming functionality of the RAG chatbot
"""
import asyncio
import json
from backend.src.services.rag_service import RAGService, RAGQuery

def test_rag_streaming():
    """Test the RAG service streaming functionality"""
    print("Testing RAG service streaming...")

    # Create a RAG service instance
    rag_service = RAGService()

    # Create a test query
    test_query = RAGQuery(
        query="What are humanoid robots?",
        mode="normal",
        session_id="test-session-123"
    )

    print("Starting streaming query...")
    try:
        # Test the streaming functionality
        for i, chunk in enumerate(rag_service.query_stream(test_query)):
            print(f"Chunk {i+1}: {chunk}")
            if i >= 10:  # Limit for testing
                break
        print("Streaming test completed successfully!")
    except Exception as e:
        print(f"Error during streaming test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rag_streaming()