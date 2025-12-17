import React from 'react';
import PropTypes from 'prop-types';
import Citation from './Citation';

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
            <Citation
              key={index}
              source={citation.source}
              content={citation.content}
              similarityScore={citation.similarity_score || citation.similarityScore}
            />
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