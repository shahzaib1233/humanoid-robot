import React, { useEffect } from 'react';
import OriginalLayout from '@theme-original/Layout';
import DocusaurusChatWidget from '../components/DocusaurusChatWidget';

export default function Layout(props) {
  return (
    <>
      <OriginalLayout {...props} />
      <DocusaurusChatWidget />
    </>
  );
}