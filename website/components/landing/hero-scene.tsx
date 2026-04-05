'use client';

import dynamic from 'next/dynamic';

const HeroSceneCanvas = dynamic(
  () => import('./hero-scene-canvas').then((mod) => mod.HeroSceneCanvas),
  { ssr: false },
);

export function HeroScene() {
  return (
    <div className="absolute inset-0 z-0">
      <HeroSceneCanvas />
    </div>
  );
}
