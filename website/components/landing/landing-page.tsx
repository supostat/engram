'use client';

import { Hero } from './hero';
import { Features } from './features';
import { HowItWorks } from './how-it-works';
import { QuickStart } from './quick-start';
import { Cta } from './cta';
import { Footer } from './footer';
import type { Lang } from './translations';

interface LandingPageProps {
  lang: Lang;
}

export function LandingPage({ lang }: LandingPageProps) {
  return (
    <main className="bg-[#0a0a1a]">
      <Hero lang={lang} />
      <Features lang={lang} />
      <HowItWorks lang={lang} />
      <QuickStart lang={lang} />
      <Cta lang={lang} />
      <Footer lang={lang} />
    </main>
  );
}
