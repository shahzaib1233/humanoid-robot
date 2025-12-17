# Quickstart Guide: RAG Chatbot

## Overview
This guide provides a quick introduction to setting up and running the Retrieval-Augmented Generation (RAG) chatbot for the humanoid robots book.

## Prerequisites
- Python 3.11+
- Node.js 18+ (for Docusaurus)
- Access to OpenRouter API
- Access to Qwen-Embedding model
- Qdrant Cloud account
- Neon Serverless PostgreSQL account

## Environment Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd <repository-name>
```

2. **Set up backend environment**:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Set up environment variables**:
Create a `.env` file in the backend directory with the following:
```env
OPENROUTER_API_KEY=your_openrouter_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=your_qdrant_cluster_url
NEON_DB_URL=your_neon_db_connection_string
QWEN_EMBEDDING_API_KEY=your_qwen_embedding_api_key
```

## Running the Application

1. **Start the backend API**:
```bash
cd backend
uvicorn main:app --reload --port 8000
```

2. **Index the book content**:
```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{
    "content_directory": "./docs"
  }'
```

3. **Test the chat endpoint**:
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key principles of humanoid robotics?",
    "mode": "normal"
  }'
```

## Frontend Integration

1. **Install Docusaurus dependencies**:
```bash
cd frontend
npm install
```

2. **Run the Docusaurus site with the chatbot widget**:
```bash
npm start
```

3. **The chatbot widget will be embedded in all documentation pages**.

## Key Features

1. **Normal Mode**: Ask questions about the book content and get answers with citations
2. **Selection-Only Mode**: Select text on a page and ask questions based only on that selection
3. **Streaming Responses**: Get real-time responses as they're generated
4. **Citations**: All responses include citations to the original book content

## API Endpoints

- `POST /index`: Process and embed Docusaurus content
- `POST /chat`: Get RAG-powered responses to user queries

## Configuration

The system can be configured through environment variables to adjust:
- Chunk size for content processing
- Embedding model parameters
- Database connection settings
- API timeout values