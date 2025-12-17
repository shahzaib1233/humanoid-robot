import React from 'react';
import PropTypes from 'prop-types';

const Citation = ({ source, content, similarityScore }) => {
  return (
    <div className="citation">
      <div className="citation-source">
        <strong>Source:</strong> {source}
      </div>
      <div className="citation-content">
        <strong>Excerpt:</strong> {content}
      </div>
      <div className="citation-score">
        <strong>Relevance:</strong> {(similarityScore * 100).toFixed(1)}%
      </div>
    </div>
  );
};

Citation.propTypes = {
  source: PropTypes.string.isRequired,
  content: PropTypes.string.isRequired,
  similarityScore: PropTypes.number.isRequired
};

export default Citation;