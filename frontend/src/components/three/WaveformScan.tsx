'use client';

import { useRef, useMemo, useEffect, useState } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';

/* ─── Simplex noise for amplitude variation ─── */
const NOISE_GLSL = `
vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) {
  const vec2 C = vec2(1.0/6.0, 1.0/3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
  vec3 i  = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);
  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);
  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;
  i = mod289(i);
  vec4 p = permute(permute(permute(
    i.z + vec4(0.0, i1.z, i2.z, 1.0))
  + i.y + vec4(0.0, i1.y, i2.y, 1.0))
  + i.x + vec4(0.0, i1.x, i2.x, 1.0));
  float n_ = 0.142857142857;
  vec3  ns = n_ * D.wyz - D.xzx;
  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);
  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);
  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);
  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));
  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;
  vec3 p0 = vec3(a0.xy,h.x);
  vec3 p1 = vec3(a0.zw,h.y);
  vec3 p2 = vec3(a1.xy,h.z);
  vec3 p3 = vec3(a1.zw,h.w);
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}
`;

const vertexShader = `
  uniform float uTime;
  uniform float uMode;
  varying vec2 vUv;
  varying float vElevation;

  void main() {
    vUv = uv;
    float elevation = 0.0;
    vec3 newPos = position;

    if (uMode < 0.5) {
      // PREVIEW MODE: Clear, big signal (oscilloscope beam)
      float freq1 = sin(vUv.x * 12.0 - uTime * 4.0) * 0.8;
      float freq2 = sin(vUv.x * 30.0 + uTime * 2.0) * 0.3;
      
      float envelope = sin(vUv.x * 3.1415);
      elevation = (freq1 + freq2) * pow(envelope, 0.5);
      newPos.y += elevation;
    } else {
      // PROCESSING MODE: Wide grid, moving to and fro
      float freq1 = sin(vUv.x * 25.0 - uTime * 3.0) * 0.25;
      float freq2 = sin(vUv.x * 60.0 + uTime * 1.5) * 0.1;
      float freq3 = sin(vUv.x * 10.0 + uTime * 0.5) * 0.5;
      
      float envelope = pow(sin(vUv.x * 3.1415) * sin(vUv.y * 3.1415), 1.5);
      elevation = (freq1 + freq2 + freq3) * envelope * 1.2;
      newPos.y += elevation;
      
      // Move to and fro effect
      newPos.x += sin(uTime * 1.5 + vUv.y * 10.0) * 0.3;
      newPos.z += cos(uTime * 1.2 + vUv.x * 15.0) * 0.15;
    }

    vElevation = elevation;
    vec4 mvPosition = modelViewMatrix * vec4(newPos, 1.0);
    
    if (uMode < 0.5) {
      gl_PointSize = (2.5 + abs(elevation) * 2.0) * (15.0 / -mvPosition.z);
    } else {
      gl_PointSize = (1.2 + abs(elevation)) * (10.0 / -mvPosition.z);
    }
    
    gl_Position = projectionMatrix * mvPosition;
  }
`;

const fragmentShader = `
  uniform float uMode;
  varying vec2 vUv;
  varying float vElevation;

  void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    if (dist > 0.5) discard;
    
    float alpha = smoothstep(0.5, 0.3, dist) * 0.8;

    vec3 baseColor = vec3(0.05, 0.35, 0.85);
    vec3 midColor = vec3(0.1, 0.7, 0.95);
    vec3 peakColor = vec3(0.8, 0.95, 1.0);

    float h = abs(vElevation);
    vec3 color = mix(baseColor, midColor, smoothstep(0.0, 0.3, h));
    color = mix(color, peakColor, smoothstep(0.4, 0.8, h));
    
    float edgeFade = 1.0;
    if (uMode > 0.5) {
      edgeFade = pow(sin(vUv.x * 3.1415) * sin(vUv.y * 3.1415), 0.8);
    } else {
      edgeFade = sin(vUv.x * 3.1415); // Only fade X for the signal ribbon
    }

    gl_FragColor = vec4(color, alpha * edgeFade);
  }
`;

export function ParticleWaveform({ freeze, mode = 'processing' }: { freeze: boolean; mode?: 'preview' | 'processing' }) {
  const pointsRef = useRef<THREE.Points>(null);

  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uMode: { value: mode === 'preview' ? 0.0 : 1.0 },
    }),
    [mode]
  );

  useFrame((state) => {
    if (!freeze) {
      uniforms.uTime.value = state.clock.getElapsedTime();
    }
  });

  const { pos, uvs, count } = useMemo(() => {
    const isPreview = mode === 'preview';
    const gridX = isPreview ? 400 : 250;
    const gridZ = isPreview ? 10 : 80;
    const count = gridX * gridZ;
    
    const posArray = new Float32Array(count * 3);
    const uvsArray = new Float32Array(count * 2);
    
    const width = isPreview ? 14.0 : 12.0;  // X span
    const depth = isPreview ? 0.5 : 4.0;    // Z span
    
    let i = 0;
    for(let x = 0; x < gridX; x++) {
      for(let z = 0; z < gridZ; z++) {
        const posX = (x / (gridX - 1)) * width - (width / 2);
        const posZ = (z / (gridZ - 1)) * depth - (depth / 2);
        
        posArray[i * 3] = posX;
        posArray[i * 3 + 1] = 0;
        posArray[i * 3 + 2] = posZ;
        
        uvsArray[i * 2] = x / (gridX - 1);
        uvsArray[i * 2 + 1] = z / (gridZ - 1);
        i++;
      }
    }
    return { pos: posArray, uvs: uvsArray, count };
  }, [mode]);

  return (
    <points ref={pointsRef} position={[0, mode === 'preview' ? 0 : -0.5, 0]} rotation={[mode === 'preview' ? 0 : Math.PI / 6, 0, 0]}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[pos, 3]}
        />
        <bufferAttribute
          attach="attributes-uv"
          args={[uvs, 2]}
        />
      </bufferGeometry>
      <shaderMaterial
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        uniforms={uniforms}
        transparent
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

const STATUS_MESSAGES = [
  'EXTRACTING 11-CHANNEL FEATURES...',
  'RUNNING SE-VOICERESNET-18 INFERENCE...',
  'CALIBRATING CONFIDENCE SCORES...',
  'ANALYZING SLIDING WINDOWS...',
  'COMPUTING SPECTRAL CONTRAST...',
  'AGGREGATING WINDOW PROBABILITIES...',
];

function generateSessionId(): string {
  const chars = '0123456789ABCDEF';
  let result = '';
  for (let i = 0; i < 8; i++) {
    result += chars[Math.floor(Math.random() * chars.length)];
  }
  return result;
}

export default function WaveformScan() {
  const [statusIndex, setStatusIndex] = useState(0);
  const [reducedMotion, setReducedMotion] = useState(false);
  const [sessionId] = useState(() => generateSessionId());

  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReducedMotion(mq.matches);
    const handler = (e: MediaQueryListEvent) => setReducedMotion(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      setStatusIndex((prev) => (prev + 1) % STATUS_MESSAGES.length);
    }, 1500);
    return () => clearInterval(interval);
  }, []);

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        width: '100vw',
        height: '100vh',
        zIndex: 50,
        background: 'var(--bg-void)',
      }}
    >
      {/* Three.js fullscreen waveform */}
      <div style={{ position: 'absolute', inset: 0 }}>
        <Canvas
          camera={{ position: [0, 0, 4], fov: 60 }}
          gl={{ alpha: true, antialias: true }}
          dpr={[1, 1.5]}
          style={{ background: 'transparent' }}
        >
          <ParticleWaveform freeze={reducedMotion} />
        </Canvas>
      </div>

      {/* Corner brackets */}
      {/* Top-left */}
      <div style={{
        position: 'absolute', top: '24px', left: '24px',
        width: '24px', height: '24px',
        borderTop: '2px solid rgba(56,189,248,0.5)',
        borderLeft: '2px solid rgba(56,189,248,0.5)',
      }} />
      {/* Top-right */}
      <div style={{
        position: 'absolute', top: '24px', right: '24px',
        width: '24px', height: '24px',
        borderTop: '2px solid rgba(56,189,248,0.5)',
        borderRight: '2px solid rgba(56,189,248,0.5)',
      }} />
      {/* Bottom-left */}
      <div style={{
        position: 'absolute', bottom: '24px', left: '24px',
        width: '24px', height: '24px',
        borderBottom: '2px solid rgba(56,189,248,0.5)',
        borderLeft: '2px solid rgba(56,189,248,0.5)',
      }} />
      {/* Bottom-right */}
      <div style={{
        position: 'absolute', bottom: '24px', right: '24px',
        width: '24px', height: '24px',
        borderBottom: '2px solid rgba(56,189,248,0.5)',
        borderRight: '2px solid rgba(56,189,248,0.5)',
      }} />

      {/* Top-left readout panel */}
      <div style={{
        position: 'absolute',
        top: '32px',
        left: '48px',
        fontFamily: 'var(--font-mono)',
        fontSize: '0.75rem',
        fontWeight: 600,
        letterSpacing: '0.1em',
        textTransform: 'uppercase',
        color: 'var(--accent-signal)',
        lineHeight: 1.8,
        zIndex: 2,
      }}>
        <div style={{ opacity: 0.7 }}>VOXGUARD AI &nbsp;// &nbsp;INFERENCE ENGINE v2.1</div>
        <div style={{ color: 'var(--text-muted)' }}>SESSION: {sessionId}</div>
        <div>
          STATUS: <span style={{ color: 'var(--state-uncertain)' }}>PROCESSING</span>
        </div>
      </div>

      {/* Status text — bottom center */}
      <div style={{
        position: 'absolute',
        bottom: '48px',
        left: '50%',
        transform: 'translateX(-50%)',
        textAlign: 'center',
        zIndex: 2,
      }}>
        <div style={{
          fontFamily: 'var(--font-mono)',
          fontSize: '1.1rem',
          color: '#F1F5F9',
          fontWeight: 700,
          letterSpacing: '0.1em',
          minHeight: '1.5em',
          marginBottom: '0.75rem',
          textShadow: '0 0 20px rgba(56, 189, 248, 0.6), 0 0 40px rgba(56, 189, 248, 0.3), 0 2px 8px rgba(0,0,0,0.8)',
          padding: '0.5rem 1.5rem',
          borderRadius: '8px',
          background: 'rgba(7, 10, 16, 0.6)',
        }}>
          {STATUS_MESSAGES[statusIndex]}
        </div>

        <div style={{
          display: 'flex',
          gap: '6px',
          justifyContent: 'center',
        }}>
          {STATUS_MESSAGES.map((_, i) => (
            <div
              key={i}
              style={{
                width: '6px',
                height: '6px',
                borderRadius: '50%',
                background: i === statusIndex ? 'var(--accent-signal)' : 'rgba(255,255,255,0.1)',
                transition: 'background 0.3s ease',
              }}
            />
          ))}
        </div>
      </div>

      {/* Bottom progress bar */}
      <div style={{
        position: 'absolute',
        bottom: 0,
        left: 0,
        width: '100%',
        height: '2px',
        background: 'rgba(255,255,255,0.05)',
        overflow: 'hidden',
        zIndex: 2,
      }}>
        <div style={{
          height: '100%',
          background: 'linear-gradient(90deg, var(--accent-signal), var(--accent-pulse))',
          animation: 'progressSweep 12s linear forwards',
        }} />
      </div>

      <style>{`
        @keyframes progressSweep {
          from { width: 0%; }
          to { width: 100%; }
        }
      `}</style>
    </div>
  );
}
