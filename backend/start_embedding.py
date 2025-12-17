"""
Script to start the embedding process for the humanoid robots book content.
"""
import sys
import os
import glob

# Add the current directory to the path
sys.path.insert(0, '.')

# Load environment
from dotenv import load_dotenv
load_dotenv()

print('Starting content indexing process...')

from src.services.rag_service import RAGService

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> list:
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

def main():
    # Initialize the RAG service
    rag_service = RAGService()

    # Directory containing the book content
    content_directory = os.path.join(os.path.dirname(__file__), '..', 'docs')

    print(f'Reading content from: {content_directory}')

    # Look for MD and MDX files in the directory and subdirectories
    patterns = [
        os.path.join(content_directory, '**', '*.md'),
        os.path.join(content_directory, '**', '*.mdx'),
        os.path.join(content_directory, '*.md'),
        os.path.join(content_directory, '*.mdx')
    ]

    content_files = []
    for pattern in patterns:
        for file_path in glob.glob(pattern, recursive=True):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    content_files.append((file_path, content))
                    print(f'Read file: {file_path} ({len(content)} characters)')
            except Exception as e:
                print(f'Error reading file {file_path}: {e}')

    if not content_files:
        print('No content files found!')
        return

    print(f'Found {len(content_files)} content files')

    # Process each file
    total_chunks_indexed = 0
    for file_path, content in content_files:
        print(f'\nProcessing: {file_path}')

        # Split content into chunks
        chunks = chunk_text(content, chunk_size=1000, overlap=100)
        print(f'Split into {len(chunks)} chunks')

        # Index each chunk
        for i, chunk in enumerate(chunks):
            print(f'  Indexing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...')

            try:
                success = rag_service.index_content(
                    content_text=chunk,
                    source_ref=file_path
                )

                if success:
                    total_chunks_indexed += 1
                    print(f'    Indexed successfully')
                else:
                    print(f'    Failed to index')
            except Exception as e:
                print(f'    Error indexing chunk: {e}')

    print(f'\nCompleted! Indexed {total_chunks_indexed} content chunks from {len(content_files)} files.')
    print('Content is now available for semantic search in the RAG chatbot.')

if __name__ == "__main__":
    main()