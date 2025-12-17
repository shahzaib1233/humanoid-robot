/**
 * API service for the RAG Chatbot frontend.
 * Handles communication with the backend API.
 */

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

class ChatAPI {
  async sendMessage(data) {
    try {
      // Create AbortController for timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
        signal: controller.signal, // Add signal for aborting
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // For streaming responses, we'll handle differently
      if (data.stream) {
        return response; // Return the response object for streaming
      }

      const result = await response.json();
      return result;
    } catch (error) {
      clearTimeout(timeoutId);

      if (error.name === 'AbortError') {
        console.error('Request timed out');
        throw new Error('Request timed out after 30 seconds');
      }

      console.error('Error sending message:', error);
      throw error;
    }
  }

  async indexContent(data) {
    try {
      const response = await fetch(`${API_BASE_URL}/index`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Error indexing content:', error);
      throw error;
    }
  }
}

export const chatAPI = new ChatAPI();