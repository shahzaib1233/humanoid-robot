import React, { useState, useEffect, useRef } from 'react';
import PropTypes from 'prop-types';
import Message from './Message';
import './ChatWidget.css';

const ChatWidget = ({ initialSelectedText = "" }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState(initialSelectedText);
  const [mode, setMode] = useState('normal'); // 'normal' or 'selection-only'
  const [isExpanded, setIsExpanded] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Scroll to bottom of messages
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSend = async () => {
    if (!inputValue.trim() || isLoading) return;

    // Add user message to chat
    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: inputValue
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Prepare the request payload
      const requestBody = {
        query: inputValue,
        mode: mode,
        stream: true, // Enable streaming
        ...(selectedText && { selected_text: selectedText })
      };

      // For streaming, we'll need to handle the response differently
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
        citations: []
      };

      // Add the initial assistant message to the chat
      setMessages(prev => [...prev, initialAssistantMessage]);

      // Handle the streaming response
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let currentContent = '';

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

              // Update the message in the state
              setMessages(prev =>
                prev.map(msg =>
                  msg.id === assistantMessageId
                    ? { ...msg, content: currentContent, citations: data.citations || [] }
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
    } catch (error) {
      console.error('Error sending message:', error);

      // Add error message to chat
      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.'
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const toggleMode = () => {
    setMode(prevMode => prevMode === 'normal' ? 'selection-only' : 'normal');
  };

  const clearChat = () => {
    setMessages([]);
  };

  return (
    <div className={`chat-widget ${isExpanded ? 'expanded' : 'collapsed'}`}>
      <div className="chat-header">
        <h3>Humanoid Robots Book Assistant</h3>
        <div className="chat-controls">
          <button
            className={`mode-toggle ${mode === 'selection-only' ? 'active' : ''}`}
            onClick={toggleMode}
            title={mode === 'selection-only'
              ? 'Selection-only mode: answers based only on selected text'
              : 'Normal mode: uses entire book content'}
          >
            {mode === 'selection-only' ? '🔒' : '📖'} {mode === 'selection-only' ? 'Selection Only' : 'Normal'}
          </button>
          <button onClick={clearChat} title="Clear chat">
            🗑️
          </button>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            title={isExpanded ? 'Collapse chat' : 'Expand chat'}
          >
            {isExpanded ? '−' : '+'}
          </button>
        </div>
      </div>

      {isExpanded && (
        <div className="chat-container">
          <div className="chat-messages">
            {messages.length === 0 ? (
              <div className="welcome-message">
                <p>Hello! I'm your Humanoid Robots Book Assistant.</p>
                <p>You can ask me questions about the book content.</p>
                {selectedText && (
                  <div className="selected-text-preview">
                    <p><strong>Selected text:</strong> {selectedText.substring(0, 100)}{selectedText.length > 100 ? '...' : ''}</p>
                  </div>
                )}
              </div>
            ) : (
              messages.map((message) => (
                <Message
                  key={message.id}
                  role={message.role}
                  content={message.content}
                  citations={message.citations}
                />
              ))
            )}
            {isLoading && (
              <Message
                role="assistant"
                content="Thinking..."
              />
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="chat-input-area">
            {mode === 'selection-only' && selectedText && (
              <div className="selected-text-indicator">
                Using selected text: "{selectedText.substring(0, 50)}{selectedText.length > 50 ? '...' : ''}"
              </div>
            )}
            <div className="input-container">
              <textarea
                ref={inputRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a question about the humanoid robots book..."
                disabled={isLoading}
                rows={3}
              />
              <button
                onClick={handleSend}
                disabled={!inputValue.trim() || isLoading}
                className="send-button"
              >
                {isLoading ? '...' : '→'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

ChatWidget.propTypes = {
  initialSelectedText: PropTypes.string
};

export default ChatWidget;