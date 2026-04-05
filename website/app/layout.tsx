import type { ReactNode } from 'react';
import type { Metadata } from 'next';
import './global.css';

export const metadata: Metadata = {
  icons: { icon: '/favicon.svg' },
};

export default function Layout({ children }: { children: ReactNode }) {
  return children;
}
