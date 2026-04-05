'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { HeroScene } from './hero-scene';
import { useTranslations, type Lang } from './translations';

interface HeroProps {
  lang: Lang;
}

export function Hero({ lang }: HeroProps) {
  const t = useTranslations(lang);

  return (
    <section className="relative flex min-h-screen flex-col overflow-hidden bg-[#0a0a1a]">
      {/* Top: 3D scene */}
      <div className="relative h-[55vh] w-full">
        <HeroScene />
        <div className="absolute inset-x-0 bottom-0 h-32 bg-gradient-to-t from-[#0a0a1a] to-transparent" />
      </div>

      {/* Bottom: Text content */}
      <div className="relative z-10 flex flex-1 flex-col items-center justify-center px-6 text-center">
        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          className="bg-gradient-to-r from-purple-400 via-purple-300 to-blue-400 bg-clip-text font-sans text-8xl font-bold tracking-tight text-transparent sm:text-9xl md:text-[10rem]"
        >
          {t.heroTitle}
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2, ease: 'easeOut' }}
          className="mt-4 max-w-lg text-xl text-purple-100 sm:text-2xl"
        >
          {t.heroSubtitle}
        </motion.p>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.35, ease: 'easeOut' }}
          className="mt-2 max-w-md text-lg text-purple-200/60 sm:text-xl"
        >
          {t.heroDescription}
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5, ease: 'easeOut' }}
          className="mt-10"
        >
          <Link
            href={`/${lang}/docs`}
            className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-purple-600 to-blue-600 px-8 py-4 text-lg font-semibold text-white shadow-lg shadow-purple-500/25 transition-all duration-300 hover:shadow-xl hover:shadow-purple-500/40 hover:brightness-110"
          >
            {t.heroButton}
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path d="M4 10H16M16 10L11 5M16 10L11 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </Link>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 0.8 }}
          className="mt-8 flex items-center gap-6 text-sm text-purple-300/40"
        >
          {t.heroStats.map((stat, index) => (
            <span key={index} className="flex items-center gap-1.5">
              <span className="h-1.5 w-1.5 rounded-full bg-purple-500/60" />
              {stat}
            </span>
          ))}
        </motion.div>
      </div>

      {/* Scroll indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1, delay: 1.2 }}
        className="pb-6 pt-4 flex flex-col items-center gap-2"
      >
        <span className="text-xs text-purple-300/30">{t.scrollHint}</span>
        <motion.div
          animate={{ y: [0, 8, 0] }}
          transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
        >
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none" className="text-purple-400/30">
            <path d="M10 4V16M10 16L5 11M10 16L15 11" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </motion.div>
      </motion.div>
    </section>
  );
}
