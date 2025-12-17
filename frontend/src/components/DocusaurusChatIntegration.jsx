import React, { useState, useEffect } from 'react';
import ChatWidget from './RAGChatbot/ChatWidget';

// A component that can be used in Docusaurus pages to embed the chat widget
const DocusaurusChatIntegration = () => {
  const [selectedText, setSelectedText] = useState('');

  useEffect(() => {
    // Add event listener to capture text selection on the page
    const handleSelection = () => {
      const selectedText = window.getSelection().toString().trim();
      if (selectedText) {
        setSelectedText(selectedText);
      }
    };

    document.addEventListener('mouseup', handleSelection);
    return () => {
      document.removeEventListener('mouseup', handleSelection);
    };
  }, []);

  return (
    <div className="docusaurus-chat-integration">
      <ChatWidget initialSelectedText={selectedText} />
    </div>
  );
};

export default DocusaurusChatIntegration;