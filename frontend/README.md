# RAG Chatbot Frontend

This is the frontend for the Retrieval-Augmented Generation (RAG) chatbot for the humanoid robots book.

## Features

- React-based chat widget
- Streaming responses display
- Text selection mode support
- Citation display
- Docusaurus integration capability

## Components

### ChatWidget
Main chat interface that can be embedded in Docusaurus pages.

### Message
Component to display individual messages with citations.

### Citation
Component to display source citations for responses.

### useChat Hook
React hook for managing chat state and API communication.

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Set up environment variables in `.env`:
   ```bash
   REACT_APP_API_BASE_URL=http://localhost:8000
   ```

3. Run the development server:
   ```bash
   npm start
   ```

## Integration with Docusaurus

The chat widget can be integrated into Docusaurus by including the `DocusaurusChatIntegration.jsx` component. This component automatically captures text selection from the page and makes it available to the chat widget.

## Environment Variables

- `REACT_APP_API_BASE_URL`: Base URL for the backend API (default: http://localhost:8000)

## Development

The frontend uses Create React App for development. Run `npm start` to start the development server with hot reloading.