import Link from 'next/link';

export default async function HomePage({
  params,
}: {
  params: Promise<{ lang: string }>;
}) {
  const { lang } = await params;
  const isRu = lang === 'ru';

  return (
    <main className="flex min-h-screen flex-col items-center justify-center">
      <h1 className="bg-gradient-to-r from-purple-400 to-blue-500 bg-clip-text text-6xl font-bold text-transparent">
        engram
      </h1>
      <p className="mt-4 text-xl text-fd-muted-foreground">
        {isRu ? 'AI-память для агентов' : 'AI memory for agents'}
      </p>
      <Link
        href={`/${lang}/docs`}
        className="mt-8 rounded-lg bg-fd-primary px-6 py-3 font-medium text-fd-primary-foreground"
      >
        {isRu ? 'Документация' : 'Documentation'}
      </Link>
    </main>
  );
}

export function generateStaticParams() {
  return [{ lang: 'ru' }, { lang: 'en' }];
}
