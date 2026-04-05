import { RootProvider } from 'fumadocs-ui/provider/next';
import { Space_Grotesk, Fira_Code } from 'next/font/google';
import type { ReactNode } from 'react';
import { i18n } from '@/lib/i18n';

const spaceGrotesk = Space_Grotesk({
  subsets: ['latin', 'latin-ext'],
  variable: '--font-sans',
});

const firaCode = Fira_Code({
  subsets: ['latin'],
  variable: '--font-mono',
});

export default async function LangLayout({
  params,
  children,
}: {
  params: Promise<{ lang: string }>;
  children: ReactNode;
}) {
  const { lang } = await params;

  return (
    <html lang={lang} suppressHydrationWarning>
      <body
        className={`${spaceGrotesk.variable} ${firaCode.variable} flex min-h-screen flex-col`}
      >
        <RootProvider
          i18n={{
            locale: lang,
            translations: {},
            ...i18n,
          }}
        >
          {children}
        </RootProvider>
      </body>
    </html>
  );
}

export function generateStaticParams() {
  return i18n.languages.map((lang) => ({ lang }));
}
