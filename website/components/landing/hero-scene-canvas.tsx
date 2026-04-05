'use client';

import { useRef, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { BrainMesh } from './brain-mesh';
import * as THREE from 'three';

function Particles({ count = 250 }: { count?: number }) {
  const meshRef = useRef<THREE.InstancedMesh>(null);

  const particles = useMemo(() => {
    const positions: [number, number, number][] = [];
    const sizes: number[] = [];
    for (let i = 0; i < count; i++) {
      positions.push([
        (Math.random() - 0.5) * 20,
        (Math.random() - 0.5) * 14,
        (Math.random() - 0.5) * 10 - 2,
      ]);
      sizes.push(0.01 + Math.random() * 0.025);
    }
    return { positions, sizes };
  }, [count]);

  useFrame(({ clock }) => {
    if (!meshRef.current) return;
    const elapsed = clock.getElapsedTime();
    const dummy = new THREE.Object3D();
    for (let i = 0; i < count; i++) {
      const [baseX, baseY, baseZ] = particles.positions[i];
      const drift = Math.sin(elapsed * 0.3 + i * 0.1) * 0.15;
      dummy.position.set(baseX + drift, baseY + Math.cos(elapsed * 0.2 + i * 0.05) * 0.1, baseZ);
      dummy.scale.setScalar(particles.sizes[i] * (1 + Math.sin(elapsed * 0.8 + i) * 0.3));
      dummy.updateMatrix();
      meshRef.current.setMatrixAt(i, dummy.matrix);
    }
    meshRef.current.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, count]}>
      <sphereGeometry args={[1, 6, 6]} />
      <meshBasicMaterial color="#8b5cf6" transparent opacity={0.4} />
    </instancedMesh>
  );
}

function MouseParallax() {
  const { camera } = useThree();
  const targetRef = useRef({ x: 0, y: 0 });

  useFrame(({ pointer }) => {
    targetRef.current.x = pointer.x * 0.3;
    targetRef.current.y = pointer.y * 0.2;
    camera.position.x += (targetRef.current.x - camera.position.x) * 0.05;
    camera.position.y += (targetRef.current.y - camera.position.y) * 0.05;
    camera.lookAt(0, 0, 0);
  });

  return null;
}

export function HeroSceneCanvas() {
  return (
    <Canvas
      camera={{ position: [0, 0, 5.5], fov: 55 }}
      style={{ background: 'transparent' }}
      gl={{ alpha: true, antialias: true }}
    >
      <ambientLight intensity={0.3} />
      <pointLight position={[5, 5, 5]} intensity={0.8} color="#8b5cf6" />
      <pointLight position={[-5, -3, 3]} intensity={0.4} color="#3b82f6" />
      <pointLight position={[0, 0, 4]} intensity={0.2} color="#a78bfa" />
      <BrainMesh />
      <Particles />
      <MouseParallax />
    </Canvas>
  );
}
