import { i18n } from '@/lib/i18n';
import { loader } from 'fumadocs-core/source';
import { docs } from '@/.source';

const mdxSource = docs.toFumadocsSource();

// fumadocs-mdx@11 returns files as a lazy getter (function),
// while fumadocs-core@15 loader expects a plain array
const files = typeof mdxSource.files === 'function'
  ? (mdxSource as unknown as { files: () => typeof mdxSource.files }).files()
  : mdxSource.files;

export const source = loader({
  i18n,
  baseUrl: '/docs',
  source: { files },
});
