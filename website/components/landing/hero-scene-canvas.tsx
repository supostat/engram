'use client';

import { Canvas } from '@react-three/fiber';
import { BrainMesh } from './brain-mesh';

export function HeroSceneCanvas() {
  return (
    <Canvas
      camera={{ position: [0, 0, 6], fov: 50 }}
      style={{ background: 'transparent' }}
      gl={{ alpha: true, antialias: true }}
    >
      <ambientLight intensity={0.3} />
      <pointLight position={[5, 5, 5]} intensity={0.8} color="#8b5cf6" />
      <pointLight position={[-5, -3, 3]} intensity={0.4} color="#3b82f6" />
      <BrainMesh />
    </Canvas>
  );
}
