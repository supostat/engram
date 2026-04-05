'use client';

import Link from 'next/link';
import { useTranslations, type Lang } from './translations';

interface FooterProps {
  lang: Lang;
}

export function Footer({ lang }: FooterProps) {
  const t = useTranslations(lang);

  return (
    <footer className="border-t border-purple-500/10 bg-[#0a0a1a] px-6 py-8">
      <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 sm:flex-row">
        <span className="text-sm text-purple-200/40">{t.footerCopyright}</span>
        <div className="flex items-center gap-6">
          <Link
            href={`/${lang}/docs`}
            className="text-sm text-purple-200/40 transition-colors hover:text-purple-200/80"
          >
            {t.footerDocs}
          </Link>
          <a
            href="https://github.com/supostat/engram"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-purple-200/40 transition-colors hover:text-purple-200/80"
          >
            {t.footerGithub}
          </a>
        </div>
      </div>
    </footer>
  );
}
