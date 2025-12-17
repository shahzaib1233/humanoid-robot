import { useState, useCallback } from 'react';

const useChat = () => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const sendMessage = useCallback(async (query, selectedText = null, mode = 'normal', stream = true) => {
    setIsLoading(true);
    setError(null);

    try {
      // Add user message to chat
      const userMessage = {
        id: Date.now(),
        role: 'user',
        content: query
      };

      setMessages(prev => [...prev, userMessage]);

      // Prepare the request payload
      const requestBody = {
        query: query,
        mode: mode,
        stream: stream,
        ...(selectedText && { selected_text: selectedText })
      };

      // For streaming, we'll need to handle the response differently
      if (stream) {
        // Using fetch API directly for streaming
        const response = await fetch(`${process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000'}/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        // Create an assistant message to update as we receive stream data
        const assistantMessageId = Date.now() + 1;
        const initialAssistantMessage = {
          id: assistantMessageId,
          role: 'assistant',
          content: '',
          citations: [],
          tokensUsed: 0
        };

        // Add the initial assistant message to the chat
        setMessages(prev => [...prev, initialAssistantMessage]);

        // Handle the streaming response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let currentContent = '';
        let currentCitations = [];
        let currentTokensUsed = 0;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          // Process each complete line
          const lines = buffer.split('\n');
          buffer = lines.pop(); // Keep incomplete line in buffer

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6)); // Remove 'data: ' prefix

                // Update the assistant message content
                currentContent = data.response || '';
                currentCitations = data.citations || [];
                currentTokensUsed = data.tokens_used || 0;

                // Update the message in the state
                setMessages(prev =>
                  prev.map(msg =>
                    msg.id === assistantMessageId
                      ? {
                          ...msg,
                          content: currentContent,
                          citations: currentCitations,
                          tokensUsed: currentTokensUsed
                        }
                      : msg
                  )
                );
              } catch (e) {
                console.error('Error parsing stream data:', e);
              }
            }
          }
        }

        // Close the reader
        reader.releaseLock();

        // Return the final message
        return {
          id: assistantMessageId,
          role: 'assistant',
          content: currentContent,
          citations: currentCitations,
          tokensUsed: currentTokensUsed
        };
      } else {
        // Non-streaming implementation
        const response = await fetch(`${process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000'}/chat`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(requestBody),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();

        // Add assistant message to chat
        const assistantMessage = {
          id: Date.now() + 1,
          role: 'assistant',
          content: result.response,
          citations: result.citations || [],
          tokensUsed: result.tokens_used
        };

        setMessages(prev => [...prev, assistantMessage]);
        return assistantMessage;
      }
    } catch (err) {
      console.error('Error in useChat sendMessage:', err);
      setError(err.message);

      // Add error message to chat
      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.'
      };

      setMessages(prev => [...prev, errorMessage]);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const clearChat = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  const updateSelectedText = useCallback((selectedText) => {
    // This could be used to update context when text is selected on the page
    console.log('Selected text updated:', selectedText);
  }, []);

  return {
    messages,
    isLoading,
    error,
    sendMessage,
    clearChat,
    updateSelectedText
  };
};

export default useChat;