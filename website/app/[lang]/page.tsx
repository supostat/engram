import { LandingPage } from '@/components/landing/landing-page';
import type { Lang } from '@/components/landing/translations';

export default async function HomePage({
  params,
}: {
  params: Promise<{ lang: string }>;
}) {
  const { lang } = await params;

  return <LandingPage lang={lang as Lang} />;
}

export function generateStaticParams() {
  return [{ lang: 'ru' }, { lang: 'en' }];
}
