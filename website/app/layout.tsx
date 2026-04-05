import { RootProvider } from 'fumadocs-ui/provider/next';
import { Space_Grotesk, Fira_Code } from 'next/font/google';
import type { ReactNode } from 'react';
import type { Metadata } from 'next';
import './global.css';

const spaceGrotesk = Space_Grotesk({
  subsets: ['latin', 'latin-ext'],
  variable: '--font-sans',
});

const firaCode = Fira_Code({
  subsets: ['latin'],
  variable: '--font-mono',
});

export const metadata: Metadata = {
  icons: { icon: '/favicon.svg' },
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="ru" suppressHydrationWarning>
      <body
        className={`${spaceGrotesk.variable} ${firaCode.variable} flex min-h-screen flex-col`}
      >
        <RootProvider>{children}</RootProvider>
      </body>
    </html>
  );
}
