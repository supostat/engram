import type { ReactNode } from 'react';
import { i18n } from '@/lib/i18n';

export default function LangLayout({ children }: { children: ReactNode }) {
  return children;
}

export function generateStaticParams() {
  return i18n.languages.map((lang) => ({ lang }));
}
