'use client';

import { motion, useInView } from 'framer-motion';
import { useRef } from 'react';
import { useTranslations, type Lang } from './translations';

interface QuickStartProps {
  lang: Lang;
}

const codeLines = [
  { prefix: '$ ', text: 'cargo install engram-core', color: 'text-green-400' },
  { prefix: '$ ', text: 'npm install -g @engram/mcp-server', color: 'text-green-400' },
  { prefix: '$ ', text: 'engram init', color: 'text-green-400' },
];

export function QuickStart({ lang }: QuickStartProps) {
  const t = useTranslations(lang);
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: '-80px' });

  return (
    <section className="bg-[#0a0a1a] px-6 py-24" ref={ref}>
      <div className="mx-auto max-w-3xl">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
          transition={{ duration: 0.5 }}
          className="mb-12 text-center text-4xl font-bold text-white"
        >
          {t.quickStartTitle}
        </motion.h2>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
          transition={{ duration: 0.5, delay: 0.15 }}
          className="overflow-hidden rounded-2xl border border-purple-500/15 bg-[#0d0d24] shadow-2xl shadow-purple-500/5"
        >
          <div className="flex items-center gap-2 border-b border-purple-500/10 px-5 py-3">
            <span className="h-3 w-3 rounded-full bg-red-500/60" />
            <span className="h-3 w-3 rounded-full bg-yellow-500/60" />
            <span className="h-3 w-3 rounded-full bg-green-500/60" />
            <span className="ml-3 text-xs text-purple-200/40">terminal</span>
          </div>

          <div className="p-6 font-mono text-sm leading-8 sm:text-base">
            {codeLines.map((line, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -10 }}
                animate={
                  isInView
                    ? { opacity: 1, x: 0 }
                    : { opacity: 0, x: -10 }
                }
                transition={{
                  duration: 0.4,
                  delay: 0.3 + index * 0.15,
                }}
              >
                <span className="text-purple-400/60">{line.prefix}</span>
                <span className={line.color}>{line.text}</span>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
