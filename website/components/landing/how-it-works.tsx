'use client';

import { motion, useInView } from 'framer-motion';
import { useRef } from 'react';
import { useTranslations, type Lang } from './translations';

interface HowItWorksProps {
  lang: Lang;
}

const stepIcons = [
  <svg key="store" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/></svg>,
  <svg key="search" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>,
  <svg key="judge" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M12 3v18M3 12h18M5.636 5.636l12.728 12.728M18.364 5.636L5.636 18.364"/></svg>,
  <svg key="learn" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>,
];

export function HowItWorks({ lang }: HowItWorksProps) {
  const t = useTranslations(lang);
  const ref = useRef<HTMLDivElement>(null);
  const isInView = useInView(ref, { once: true, margin: '-80px' });

  const steps = [
    { title: t.stepStore, description: t.stepStoreDescription },
    { title: t.stepSearch, description: t.stepSearchDescription },
    { title: t.stepJudge, description: t.stepJudgeDescription },
    { title: t.stepLearn, description: t.stepLearnDescription },
  ];

  return (
    <section className="bg-[#0a0a1a] px-6 py-24" ref={ref}>
      <div className="mx-auto max-w-5xl">
        <motion.h2
          initial={{ opacity: 0, y: 20 }}
          animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
          transition={{ duration: 0.5 }}
          className="mb-20 text-center text-4xl font-bold text-white"
        >
          {t.howItWorksTitle}
        </motion.h2>

        <div className="relative flex flex-col items-center gap-12 md:flex-row md:gap-0">
          <div className="absolute left-8 top-0 hidden h-full w-px bg-gradient-to-b from-purple-500/0 via-purple-500/30 to-purple-500/0 md:left-0 md:top-1/2 md:h-px md:w-full md:bg-gradient-to-r md:block" />

          {steps.map((step, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 30 }}
              animate={
                isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 30 }
              }
              transition={{
                duration: 0.5,
                delay: 0.15 * index,
                ease: 'easeOut',
              }}
              className="relative flex-1 px-4 text-center"
            >
              <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl border border-purple-500/20 bg-purple-500/10 text-purple-400">
                {stepIcons[index]}
              </div>
              <h3 className="mb-2 text-lg font-semibold text-white">
                {step.title}
              </h3>
              <p className="text-sm leading-relaxed text-purple-200/60">
                {step.description}
              </p>

              {index < steps.length - 1 && (
                <div className="absolute -bottom-8 left-1/2 -translate-x-1/2 text-purple-500/40 md:-right-4 md:bottom-auto md:left-auto md:top-8 md:translate-x-0">
                  <svg
                    width="20"
                    height="20"
                    viewBox="0 0 20 20"
                    fill="none"
                    className="rotate-90 md:rotate-0"
                  >
                    <path
                      d="M4 10H16M16 10L11 5M16 10L11 15"
                      stroke="currentColor"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </div>
              )}
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
