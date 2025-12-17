import React from 'react';
import DocusaurusChatWidget from './index';

// Root component that will wrap the entire Docusaurus app
const Root = ({ children }) => {
  return (
    <>
      {children}
      <DocusaurusChatWidget />
    </>
  );
};

export default Root;