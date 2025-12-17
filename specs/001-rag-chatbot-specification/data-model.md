# Data Model: RAG Chatbot

## Core Entities

### Book Content Chunk
Represents a segment of book content that has been processed and embedded for retrieval

**Fields**:
- `id` (UUID): Unique identifier for the chunk
- `content_text` (TEXT): The actual text content of the chunk
- `embedding_vector` (VECTOR): The embedding vector representation of the content
- `source_document_ref` (VARCHAR): Reference to the original document (file path, URL, etc.)
- `metadata` (JSONB): Additional metadata (section, chapter, page, etc.)
- `created_at` (TIMESTAMP): Timestamp when the chunk was created
- `updated_at` (TIMESTAMP): Timestamp when the chunk was last updated

### Chat Session
Represents a conversation between a user and the chatbot

**Fields**:
- `id` (UUID): Unique identifier for the session
- `user_id` (UUID): Identifier for the user (could be anonymous UUID)
- `session_title` (VARCHAR): Brief title for the session
- `created_at` (TIMESTAMP): Timestamp when the session was created
- `updated_at` (TIMESTAMP): Timestamp when the session was last updated
- `is_active` (BOOLEAN): Whether the session is currently active

### User Query
Represents a question or request from a user

**Fields**:
- `id` (UUID): Unique identifier for the query
- `chat_session_id` (UUID): Reference to the chat session
- `query_text` (TEXT): The actual text of the user's query
- `selected_text` (TEXT): Text selected by the user on the page (nullable)
- `mode` (ENUM): Query mode ('normal' or 'selection-only')
- `timestamp` (TIMESTAMP): When the query was submitted
- `metadata` (JSONB): Additional metadata about the query

### Retrieval Result
Represents the relevant content chunks retrieved from the vector database based on user query

**Fields**:
- `id` (UUID): Unique identifier for the retrieval result
- `user_query_id` (UUID): Reference to the user query
- `chunk_id` (UUID): Reference to the book content chunk
- `similarity_score` (FLOAT): Similarity score from vector search (0.0-1.0)
- `rank` (INTEGER): Rank position in the retrieval results
- `timestamp` (TIMESTAMP): When the retrieval was performed

### Generated Response
Represents the LLM-generated answer to a user query

**Fields**:
- `id` (UUID): Unique identifier for the response
- `user_query_id` (UUID): Reference to the user query
- `response_text` (TEXT): The generated response text
- `citations` (JSONB): Citations to source material
- `confidence_indicator` (FLOAT): Confidence score of the response (0.0-1.0)
- `timestamp` (TIMESTAMP): When the response was generated
- `token_usage` (JSONB): Information about token usage for the generation

## Relationships

1. **One-to-Many**: Chat Session → User Query (one session can have many queries)
2. **One-to-Many**: User Query → Retrieval Result (one query can retrieve multiple chunks)
3. **One-to-One**: User Query → Generated Response (one query generates one response)
4. **One-to-Many**: Book Content Chunk → Retrieval Result (one chunk can be retrieved multiple times)

## Constraints

1. **Foreign Key Constraints**: All relationship references must exist
2. **Check Constraints**:
   - Similarity scores must be between 0.0 and 1.0
   - Confidence indicators must be between 0.0 and 1.0
   - Mode must be either 'normal' or 'selection-only'

## Indexes

1. **Content Chunks**: Index on `source_document_ref` for efficient document lookup
2. **Chat Sessions**: Index on `user_id` and `is_active` for session management
3. **User Queries**: Index on `chat_session_id` and `timestamp` for chronological query retrieval
4. **Retrieval Results**: Index on `user_query_id` and `rank` for ordered results retrieval
5. **Embedding Vector**: Special vector index on `embedding_vector` for similarity search