# Feature Specification: Retrieval-Augmented Generation (RAG) Chatbot

**Feature Branch**: `001-rag-chatbot-specification`
**Created**: 2025-12-16
**Status**: Draft
**Input**: User description: "Define the technical specifications for the Retrieval-Augmented Generation (RAG) chatbot.

Core Requirements:

1. **Frontend (Docusaurus):**
   Embed a custom React-based chatbot component/widget inside the Docusaurus site.
   The widget must support streaming responses and user text selection from book pages.

2. **API Backend:**
   Use **FastAPI** to create a REST API with endpoints for:
   - `/index`: Process and embed Docusaurus content (MD/MDX files).
   - `/chat`: Accept user queries and return RAG-powered responses.

3. **Embedding Model:**
   Use the **Qwen-Embedding** model for text vectorization and semantic search.

4. **Vector Database:**
   Use **Qdrant Cloud (Free Tier)** for high-performance vector similarity search and retrieval.

5. **LLM / Generator:**
   Use **OpenRouter API** to access a fast, capable large language model for response generation.

6. **Database:**
   Use **Neon Serverless PostgreSQL** for storing book metadata, content chunks, chat history,
   retrieval traces, and citation mappings.

7. **Key Feature – Selection-Only Mode:**
   The `/chat` endpoint must support RAG based on user-selected text.
   When selection-only mode is enabled, the chatbot must answer using ONLY the selected text.
   If the selected text is insufficient, the response must be:
   \"I don't have enough info in the selected text.\"

8. **Citation Enforcement:**
   All responses must include citations to the original book content."

## User Scenarios & Testing *(mandatory)*

<!--
  IMPORTANT: User stories should be PRIORITIZED as user journeys ordered by importance.
  Each user story/journey must be INDEPENDENTLY TESTABLE - meaning if you implement just ONE of them,
  you should still have a viable MVP (Minimum Viable Product) that delivers value.

  Assign priorities (P1, P2, P3, etc.) to each story, where P1 is the most critical.
  Think of each story as a standalone slice of functionality that can be:
  - Developed independently
  - Tested independently
  - Deployed independently
  - Demonstrated to users independently
-->

### User Story 1 - Interactive Book Q&A (Priority: P1)

As a reader of the humanoid robots book, I want to ask questions about the content and get accurate answers with citations, so I can better understand complex concepts and find relevant information quickly.

**Why this priority**: This is the core value proposition of the RAG chatbot - enabling users to interact with the book content through natural language queries.

**Independent Test**: Can be fully tested by asking a question about the book content and verifying that the response is accurate, relevant, and includes proper citations to the source material.

**Acceptance Scenarios**:

1. **Given** I am viewing the humanoid robots book on the Docusaurus site, **When** I type a question in the chat widget, **Then** I receive a relevant answer with citations to the original book content
2. **Given** I have selected specific text on a book page, **When** I ask a question in selection-only mode, **Then** the response is based only on the selected text with appropriate citations

---

### User Story 2 - Content Indexing and Search (Priority: P2)

As a content administrator, I want the system to automatically index new or updated book content, so that the chatbot has access to the most current information.

**Why this priority**: Without proper indexing, the chatbot cannot provide accurate answers to user questions, making this essential for maintaining quality.

**Independent Test**: Can be tested by adding new content to the book, triggering the indexing process, and verifying that the new content is searchable by the chatbot.

**Acceptance Scenarios**:

1. **Given** new MD/MDX content has been added to the Docusaurus site, **When** the indexing endpoint is called, **Then** the content is processed and embedded in the vector database

---

### User Story 3 - Streaming Responses (Priority: P3)

As a user, I want to see the chatbot's response as it's being generated rather than waiting for the full response, so I can get partial information quickly and have a more natural conversation experience.

**Why this priority**: While not essential for basic functionality, streaming responses improve user experience significantly by providing immediate feedback.

**Independent Test**: Can be tested by asking a question and observing that the response appears incrementally rather than all at once.

**Acceptance Scenarios**:

1. **Given** I have asked a question in the chat widget, **When** the response is being generated, **Then** I see the text appear progressively in real-time

---

### Edge Cases

- What happens when a user asks a question that has no relevant information in the book content?
- How does the system handle malformed or malicious user input?
- What happens when the vector database is temporarily unavailable?
- How does the system handle very long text selections in selection-only mode?
- What happens when the selected text is insufficient to answer the question in selection-only mode?
- How does the system handle concurrent users asking questions simultaneously?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a chatbot widget embedded in the Docusaurus site that allows users to ask questions about the book content
- **FR-002**: System MUST support streaming responses from the chatbot to provide real-time feedback to users
- **FR-003**: System MUST allow users to select text on book pages and use that selection as context for their questions
- **FR-004**: System MUST provide an `/index` API endpoint that processes Docusaurus MD/MDX content files and creates semantic representations
- **FR-005**: System MUST provide a `/chat` API endpoint that accepts user queries and returns RAG-powered responses
- **FR-006**: System MUST use the Qwen-Embedding model for text vectorization and semantic search
- **FR-007**: System MUST store content embeddings in a vector database (Qdrant Cloud) for similarity search to enable semantic retrieval
- **FR-008**: System MUST use OpenRouter API to access a large language model for response generation
- **FR-009**: System MUST store book metadata, content chunks, chat history, retrieval traces, and citation mappings in Neon Serverless PostgreSQL
- **FR-010**: System MUST support selection-only mode where responses are generated based ONLY on user-selected text
- **FR-011**: System MUST return "I don't have enough info in the selected text" when selected text is insufficient to answer the question in selection-only mode
- **FR-012**: System MUST include citations to original book content in all generated responses
- **FR-013**: System MUST handle concurrent user requests without degradation in performance

### Key Entities

- **Book Content Chunk**: Represents a segment of book content that has been processed and embedded for retrieval; includes content text, embedding vector, source document reference, and metadata
- **Chat Session**: Represents a conversation between a user and the chatbot; includes user queries, system responses, timestamps, and session context
- **User Query**: Represents a question or request from a user; includes the query text, selected text (if any), mode (normal or selection-only), and metadata
- **Retrieval Result**: Represents the relevant content chunks retrieved from the vector database based on user query; includes similarity scores and source citations
- **Generated Response**: Represents the LLM-generated answer to a user query; includes response text, citations to source material, and confidence indicators

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can ask questions about book content and receive relevant, accurate answers with proper citations within 5 seconds
- **SC-002**: The system can handle many users asking questions simultaneously without significant response time degradation
- **SC-003**: 90% of user queries result in responses that contain accurate information with proper citations to the source material
- **SC-004**: The system can process a large volume of book content within a reasonable time frame
- **SC-005**: Users can select text on book pages and use it as context for questions with high accuracy in the system's understanding of the selected context
- **SC-006**: When selection-only mode is enabled and selected text is insufficient, the system correctly responds with "I don't have enough info in the selected text" 100% of the time
- **SC-007**: 95% of generated responses include proper citations to the original book content
- **SC-008**: The system maintains high availability during business hours
- **SC-009**: Users rate the chatbot's helpfulness with an average score of 4.0 or higher out of 5.0
- **SC-010**: The system successfully answers 80% of questions that are directly answerable from the book content

## Assumptions and Dependencies

- The Docusaurus site is properly structured with MD/MDX content files
- The book content is available in a digital format suitable for processing
- Users have standard web browsers that support JavaScript and React components
- External APIs (Qwen-Embedding, OpenRouter, Qdrant Cloud, Neon PostgreSQL) are available and accessible
- The system has internet connectivity to access cloud services
- Book content updates will trigger re-indexing as needed
