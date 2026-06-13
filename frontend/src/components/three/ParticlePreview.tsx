'use client';

import { Canvas } from '@react-three/fiber';
import { ParticleWaveform } from './WaveformScan';

export default function ParticlePreview({ freeze = false }: { freeze?: boolean }) {
  return (
    <div style={{ position: 'absolute', inset: 0 }}>
      <Canvas
        camera={{ position: [0, 0, 5], fov: 35 }}
        gl={{ alpha: true, antialias: true }}
        dpr={[1, 1.5]}
        style={{ background: 'transparent' }}
      >
        <ParticleWaveform freeze={freeze} mode="preview" />
      </Canvas>
    </div>
  );
}
