# Implementation Plan: RAG Chatbot

**Branch**: `001-rag-chatbot-specification` | **Date**: 2025-12-16 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-rag-chatbot-specification/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implementation of a Retrieval-Augmented Generation (RAG) chatbot for the humanoid robots book. The system consists of a FastAPI backend with Qwen-Embedding for semantic search, Qdrant Cloud for vector storage, OpenRouter API for response generation, and Neon Serverless PostgreSQL for metadata storage. A React-based widget is embedded in the Docusaurus site to provide an interactive chat interface with streaming responses and text selection functionality.

## Technical Context

**Language/Version**: Python 3.11, JavaScript/TypeScript for frontend components
**Primary Dependencies**: FastAPI, Pydantic, Qwen-Embedding SDK, Qdrant client, SQLAlchemy, React
**Storage**: Neon Serverless PostgreSQL, Qdrant Cloud vector database
**Testing**: pytest for backend, Jest for frontend
**Target Platform**: Linux server (backend), Web browsers (frontend)
**Project Type**: Web application with separate frontend and backend
**Performance Goals**: <2s response time for chat queries, <5s for indexing operations
**Constraints**: <200ms p95 latency for API responses, support for concurrent users, streaming responses
**Scale/Scope**: Support for multiple concurrent users, handle large book content (40k-50k words)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Comprehensiveness**: The RAG implementation ensures comprehensive coverage of book content through vector search
- **Clarity and Accessibility**: The chatbot interface provides an accessible way to interact with complex content
- **Reproducibility and Verification**: All API endpoints and data models are well-documented with contracts
- **Evidence-Based Content**: Citations in responses link back to original book content
- **Engagement and Practical Application**: Interactive Q&A enhances engagement with the textbook
- **Visual Learning Support**: The chat interface provides clear visual feedback and citations

## Project Structure

### Documentation (this feature)

```text
specs/[###-feature]/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── models/
│   │   ├── book_content_chunk.py
│   │   ├── chat_session.py
│   │   ├── user_query.py
│   │   ├── retrieval_result.py
│   │   └── generated_response.py
│   ├── services/
│   │   ├── embedding_service.py
│   │   ├── vector_db_service.py
│   │   ├── llm_service.py
│   │   └── rag_service.py
│   ├── api/
│   │   ├── endpoints/
│   │   │   ├── index.py
│   │   │   └── chat.py
│   │   └── main.py
│   └── database/
│       ├── connection.py
│       └── migrations.py
└── tests/
    ├── unit/
    ├── integration/
    └── contract/

frontend/
├── src/
│   ├── components/
│   │   └── RAGChatbot/
│   │       ├── ChatWidget.jsx
│   │       ├── Message.jsx
│   │       └── Citation.jsx
│   ├── services/
│   │   └── api.js
│   └── hooks/
│       └── useChat.js
└── tests/
    ├── unit/
    └── integration/

docs/
└── ...
```

**Structure Decision**: Web application with separate backend and frontend components to allow independent scaling and development. The backend handles all RAG processing, while the frontend provides the user interface embedded in the Docusaurus site.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [None] | [N/A] | [N/A] |
