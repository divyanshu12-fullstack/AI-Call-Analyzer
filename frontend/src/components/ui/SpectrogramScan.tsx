'use client';

import { useEffect, useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';

interface SpectrogramScanProps {
  verdict: 'AI_GENERATED' | 'HUMAN';
}

export default function SpectrogramScan({ verdict }: SpectrogramScanProps) {
  const shouldReduceMotion = useReducedMotion();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const isAi = verdict === 'AI_GENERATED';
  const color = isAi ? 'var(--state-ai)' : 'var(--state-human)';
  const columns = 28;
  const rows = 8;

  // Generate random seeds for initial heights
  const [heights, setHeights] = useState<number[]>([]);

  useEffect(() => {
    setHeights(
      Array.from({ length: columns }).map(() =>
        Math.floor(Math.random() * (rows - 2) + 2)
      )
    );

    if (shouldReduceMotion) return;

    const interval = setInterval(() => {
      setHeights((prev) =>
        prev.map((h) => {
          const delta = Math.random() > 0.5 ? 1 : -1;
          return Math.max(1, Math.min(rows, h + delta));
        })
      );
    }, 150);

    return () => clearInterval(interval);
  }, [shouldReduceMotion]);

  if (!mounted || heights.length === 0) {
    return (
      <div style={{ height: '120px', background: 'rgba(255,255,255,0.01)' }} />
    );
  }

  return (
    <div
      className="glass-panel"
      style={{
        padding: '1.25rem',
        width: '100%',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.75rem',
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <span
          className="text-label"
          style={{
            fontSize: '0.7rem',
            letterSpacing: '0.12em',
          }}
        >
          Spectral Energy Footprint
        </span>
        <span
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '0.65rem',
            color,
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
          }}
        >
          {isAi ? 'anomaly signature active' : 'natural signature verified'}
        </span>
      </div>

      {/* Grid container */}
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-end',
          justifyContent: 'space-between',
          height: '90px',
          padding: '0.5rem 0.25rem',
          background: 'rgba(0, 0, 0, 0.2)',
          borderRadius: '6px',
          border: '1px solid var(--border-subtle)',
          overflow: 'hidden',
        }}
      >
        {heights.map((h, colIdx) => (
          <div
            key={colIdx}
            style={{
              display: 'flex',
              flexDirection: 'column-reverse',
              gap: '2px',
              width: `${100 / columns - 1}%`,
              height: '100%',
              justifyContent: 'flex-start',
            }}
          >
            {Array.from({ length: rows }).map((_, rowIdx) => {
              const active = rowIdx < h;
              // Add a bit of gradient effect based on row height
              const opacity = active
                ? isAi
                  ? 0.3 + (rowIdx / rows) * 0.7 // Red glows stronger at high frequencies
                  : 0.8 - (rowIdx / rows) * 0.5 // Green fades out at high frequencies (typical human natural speech rolloff)
                : 0.05;

              return (
                <div
                  key={rowIdx}
                  style={{
                    height: `${100 / rows - 2}%`,
                    width: '100%',
                    background: active ? color : 'rgba(255, 255, 255, 0.1)',
                    opacity,
                    borderRadius: '1px',
                    transition: 'background-color 0.2s, opacity 0.2s',
                    boxShadow: active ? `0 0 3px ${color}` : 'none',
                  }}
                />
              );
            })}
          </div>
        ))}
      </div>

      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.6rem',
          color: 'var(--text-muted)',
        }}
      >
        <span>0 Hz (Pitch)</span>
        <span>4 kHz (Formants)</span>
        <span>8 kHz+ (Sibilance)</span>
      </div>
    </div>
  );
}
