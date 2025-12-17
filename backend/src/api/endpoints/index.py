"""
Index endpoint for the RAG Chatbot application.
Processes and embeds Docusaurus content.
"""
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from slowapi import Limiter
from slowapi.util import get_remote_address
from src.services.rag_service import RAGService
import os
import glob


# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

router = APIRouter()

# Initialize RAG service
rag_service = RAGService()


class IndexRequest(BaseModel):
    content_directory: str
    chunk_size: Optional[int] = 1000
    overlap: Optional[int] = 100


class IndexResponse(BaseModel):
    status: str
    processed_files: int
    indexed_chunks: int
    message: str


def read_content_files(directory: str) -> list:
    """
    Read content from MD/MDX files in the specified directory.

    Args:
        directory: Path to the directory containing content files

    Returns:
        List of tuples containing (file_path, content)
    """
    content_files = []

    # Look for MD and MDX files in the directory and subdirectories
    patterns = [
        os.path.join(directory, "**", "*.md"),
        os.path.join(directory, "**", "*.mdx"),
        os.path.join(directory, "*.md"),
        os.path.join(directory, "*.mdx")
    ]

    for pattern in patterns:
        for file_path in glob.glob(pattern, recursive=True):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    content_files.append((file_path, content))
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

    return content_files


def chunk_text(text: str, chunk_size: int, overlap: int) -> list:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters

    Returns:
        List of text chunks
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        # Move start by chunk_size - overlap to create overlap
        start = end - overlap

        # If we're near the end, make sure we get the last bit
        if start >= len(text) - overlap:
            break

    # Remove empty chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]

    return chunks


@router.post("/index", response_model=IndexResponse)
@limiter.limit("5/hour")  # 5 requests per hour per IP
async def index_content(request: Request, index_request: IndexRequest):
    """
    Process and embed Docusaurus content for semantic search.

    Args:
        request: Index request containing the content directory and parameters

    Returns:
        IndexResponse with processing results
    """
    try:
        # Verify directory exists
        if not os.path.isdir(index_request.content_directory):
            raise HTTPException(status_code=400, detail="Content directory does not exist")

        # Read content files
        content_files = read_content_files(index_request.content_directory)

        if not content_files:
            raise HTTPException(status_code=400, detail="No MD/MDX files found in the specified directory")

        total_chunks_indexed = 0

        # Process each file
        for file_path, content in content_files:
            # Split content into chunks
            chunks = chunk_text(content, index_request.chunk_size, index_request.overlap)

            # Index each chunk
            for chunk in chunks:
                success = rag_service.index_content(
                    content_text=chunk,
                    source_ref=file_path
                )

                if success:
                    total_chunks_indexed += 1

        return IndexResponse(
            status="success",
            processed_files=len(content_files),
            indexed_chunks=total_chunks_indexed,
            message=f"Successfully processed {len(content_files)} files into {total_chunks_indexed} chunks"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing index request: {str(e)}")