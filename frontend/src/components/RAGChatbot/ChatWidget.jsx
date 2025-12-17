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
  const [selectedModel, setSelectedModel] = useState('qwen'); // Default to Qwen
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
      // Prepare the request payload - temporarily disable streaming to simplify
      const requestBody = {
        query: inputValue,
        mode: mode,
        stream: false, // Disable streaming for now to ensure basic functionality
        model: selectedModel, // Include the selected model
        ...(selectedText && { selected_text: selectedText })
      };

      // Make the API call
      const response = await fetch(`${process.env.REACT_APP_API_BASE_URL || 'http://localhost:8080'}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // For non-streaming response, just get the JSON
      const data = await response.json();

      // Add the assistant response to the chat
      const assistantMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: data.response,
        citations: data.citations || []
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);

      // Add error message to chat
      let errorMessageContent = 'Sorry, I encountered an error. Please try again.';
      if (error.message.includes('fetch') || error.message.includes('network')) {
        errorMessageContent = 'Cannot connect to the server. Please make sure the backend is running on http://localhost:8080';
      } else if (error.name === 'AbortError' || error.message.includes('timeout')) {
        errorMessageContent = 'Request timed out. Please try again with a shorter query.';
      }

      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: errorMessageContent
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
      {/* Floating chat button when collapsed */}
      {!isExpanded && (
        <button
          className="chat-toggle-button"
          onClick={() => setIsExpanded(true)}
          title="Open chat assistant"
        >
          <div className="chat-icon">🤖</div>
        </button>
      )}

      {/* Full chat interface when expanded */}
      {isExpanded && (
        <>
          <div className="chat-header">
            <h3>Humanoid Robots Assistant</h3>
            <div className="chat-controls">
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="model-selector"
                title="Select AI model"
              >
                <option value="qwen">🤖 Qwen</option>
                <option value="gemini">🔍 Gemini</option>
                <option value="openrouter">🌐 OpenRouter</option>
              </select>
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
                onClick={() => setIsExpanded(false)}
                title="Collapse chat"
              >
                −
              </button>
            </div>
          </div>

          <div className="chat-container">
            <div className="chat-messages">
              {messages.length === 0 ? (
                <div className="welcome-message">
                  <div className="welcome-icon">🤖</div>
                  <h4>Hello! I'm your Humanoid Robots Assistant</h4>
                  <p>Choose an AI model from the dropdown to start chatting.</p>
                  <div className="model-info">
                    <p><strong>Qwen:</strong> Advanced Chinese and English model</p>
                    <p><strong>Gemini:</strong> Google's multimodal AI</p>
                    <p><strong>OpenRouter:</strong> Free open-source model</p>
                  </div>
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
                  placeholder="Ask a question about humanoid robots..."
                  disabled={isLoading}
                  rows={3}
                />
                <button
                  onClick={handleSend}
                  disabled={!inputValue.trim() || isLoading}
                  className="send-button"
                >
                  {isLoading ? '...' : '➤'}
                </button>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

ChatWidget.propTypes = {
  initialSelectedText: PropTypes.string
};

export default ChatWidget;