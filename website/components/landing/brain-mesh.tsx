'use client';

import { useRef, useMemo, useCallback } from 'react';
import { useFrame } from '@react-three/fiber';
import { Float } from '@react-three/drei';
import type { Group, BufferGeometry, BufferAttribute as ThreeBufferAttribute } from 'three';
import * as THREE from 'three';

interface Node {
  position: THREE.Vector3;
}

interface Connection {
  start: number;
  end: number;
}

function generateBrainNodes(count: number): Node[] {
  const nodes: Node[] = [];
  let attempts = 0;
  while (nodes.length < count && attempts < count * 10) {
    attempts++;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    const radiusX = 3.4;
    const radiusY = 2.6;
    const radiusZ = 2.4;
    const jitter = 0.75 + Math.random() * 0.25;
    const x = radiusX * Math.sin(phi) * Math.cos(theta) * jitter;
    const y = radiusY * Math.sin(phi) * Math.sin(theta) * jitter;
    const z = radiusZ * Math.cos(phi) * jitter;
    nodes.push({ position: new THREE.Vector3(x, y, z) });
  }
  return nodes;
}

function generateConnections(
  nodes: Node[],
  maxDistance: number,
): Connection[] {
  const connections: Connection[] = [];
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const distance = nodes[i].position.distanceTo(nodes[j].position);
      if (distance < maxDistance) {
        connections.push({ start: i, end: j });
      }
    }
  }
  return connections;
}

export function BrainMesh() {
  const groupRef = useRef<Group>(null);
  const linesRef = useRef<BufferGeometry>(null);
  const pulsePhaseRef = useRef(0);

  const { nodes, connections, linePositions, lineColors } = useMemo(() => {
    const generatedNodes = generateBrainNodes(65);
    const generatedConnections = generateConnections(generatedNodes, 2.2);

    const positions = new Float32Array(generatedConnections.length * 6);
    const colors = new Float32Array(generatedConnections.length * 6);

    generatedConnections.forEach((connection, index) => {
      const startNode = generatedNodes[connection.start];
      const endNode = generatedNodes[connection.end];
      const offset = index * 6;
      positions[offset] = startNode.position.x;
      positions[offset + 1] = startNode.position.y;
      positions[offset + 2] = startNode.position.z;
      positions[offset + 3] = endNode.position.x;
      positions[offset + 4] = endNode.position.y;
      positions[offset + 5] = endNode.position.z;

      const baseColor = [0.4, 0.2, 0.8];
      colors[offset] = baseColor[0];
      colors[offset + 1] = baseColor[1];
      colors[offset + 2] = baseColor[2];
      colors[offset + 3] = baseColor[0];
      colors[offset + 4] = baseColor[1];
      colors[offset + 5] = baseColor[2];
    });

    return {
      nodes: generatedNodes,
      connections: generatedConnections,
      linePositions: positions,
      lineColors: colors,
    };
  }, []);

  const animateConnections = useCallback(
    (elapsed: number) => {
      if (!linesRef.current) return;
      const colorAttribute = linesRef.current.getAttribute(
        'color',
      ) as ThreeBufferAttribute | null;
      if (!colorAttribute) return;

      const colors = colorAttribute.array as Float32Array;
      const pulseSpeed = 0.8;
      const pulseCount = 5;

      for (let i = 0; i < connections.length; i++) {
        const pulseIndex = i % pulseCount;
        const phase =
          (elapsed * pulseSpeed + (pulseIndex / pulseCount) * Math.PI * 2) %
          (Math.PI * 2);
        const pulse = Math.max(0, Math.sin(phase)) * 0.6;
        const offset = i * 6;

        const r = 0.4 + pulse * 0.4;
        const g = 0.2 + pulse * 0.3;
        const b = 0.8 + pulse * 0.2;

        colors[offset] = r;
        colors[offset + 1] = g;
        colors[offset + 2] = b;
        colors[offset + 3] = r;
        colors[offset + 4] = g;
        colors[offset + 5] = b;
      }
      colorAttribute.needsUpdate = true;
    },
    [connections],
  );

  useFrame(({ clock }) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = clock.getElapsedTime() * 0.12;
    }
    pulsePhaseRef.current = clock.getElapsedTime();
    animateConnections(clock.getElapsedTime());
  });

  return (
    <Float speed={1.5} rotationIntensity={0.15} floatIntensity={0.3}>
      <group ref={groupRef}>
        {nodes.map((node, index) => (
          <mesh key={index} position={node.position}>
            <sphereGeometry args={[0.07, 12, 12]} />
            <meshStandardMaterial
              color="#a78bfa"
              emissive="#8b5cf6"
              emissiveIntensity={1.8}
              toneMapped={false}
            />
          </mesh>
        ))}
        <lineSegments>
          <bufferGeometry ref={linesRef}>
            <bufferAttribute
              attach="attributes-position"
              args={[linePositions, 3]}
            />
            <bufferAttribute
              attach="attributes-color"
              args={[lineColors, 3]}
            />
          </bufferGeometry>
          <lineBasicMaterial vertexColors transparent opacity={0.35} />
        </lineSegments>
      </group>
    </Float>
  );
}
