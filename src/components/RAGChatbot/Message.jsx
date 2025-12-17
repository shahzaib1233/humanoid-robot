import React from 'react';
import PropTypes from 'prop-types';

const Message = ({ role, content, citations }) => {
  const isUser = role === 'user';
  const isAssistant = role === 'assistant';

  return (
    <div className={`message ${isUser ? 'user-message' : 'assistant-message'}`}>
      <div className={`message-content ${isUser ? 'user' : 'assistant'}`}>
        {content}
      </div>

      {isAssistant && citations && citations.length > 0 && (
        <div className="citations">
          <h4>Citations:</h4>
          {citations.map((citation, index) => (
            <div key={index} className="citation">
              <div className="citation-source">
                <strong>Source:</strong> {citation.source}
              </div>
              <div className="citation-content">
                <strong>Excerpt:</strong> {citation.content}
              </div>
              <div className="citation-score">
                <strong>Relevance:</strong> {(citation.similarity_score * 100).toFixed(1)}%
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

Message.propTypes = {
  role: PropTypes.oneOf(['user', 'assistant']).isRequired,
  content: PropTypes.string.isRequired,
  citations: PropTypes.array
};

Message.defaultProps = {
  citations: []
};

export default Message;