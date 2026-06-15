'use client';

import { useEffect, useState } from 'react';
import { motion, useReducedMotion } from 'framer-motion';

interface ConfidenceRingProps {
  value: number; // 0 to 1
  verdict: 'AI_GENERATED' | 'HUMAN';
}

export default function ConfidenceRing({ value, verdict }: ConfidenceRingProps) {
  const shouldReduceMotion = useReducedMotion();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const percentage = Math.round(value * 100);
  const size = 200;
  const strokeWidth = 8;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const fillLength = circumference * value;
  const dashOffset = circumference - fillLength;

  // Color based on confidence level
  const getRingColor = () => {
    if (verdict === 'AI_GENERATED') {
      return percentage >= 80 ? 'var(--state-ai)' : 'var(--state-uncertain)';
    } else {
      return percentage >= 80 ? 'var(--state-human)' : 'var(--state-uncertain)';
    }
  };

  const ringColor = getRingColor();

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '0.5rem',
      }}
    >
      <div style={{ position: 'relative', width: size, height: size }}>
        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          style={{ transform: 'rotate(-90deg)' }}
        >
          {/* Background ring */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="var(--border-subtle)"
            strokeWidth={strokeWidth}
          />

          {/* Progress ring */}
          <motion.circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={ringColor}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: shouldReduceMotion ? dashOffset : circumference }}
            animate={{ strokeDashoffset: mounted ? dashOffset : circumference }}
            transition={
              shouldReduceMotion
                ? { duration: 0 }
                : { duration: 1.2, ease: 'easeOut', delay: 0.3 }
            }
            style={{
              filter: `drop-shadow(0 0 6px ${ringColor})`,
            }}
          />
        </svg>

        {/* Center content */}
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '2.5rem',
              fontWeight: 700,
              color: 'var(--text-primary)',
              lineHeight: 1,
            }}
          >
            {percentage}%
          </span>
          <span
            className="text-label"
            style={{
              color: 'var(--text-secondary)',
              marginTop: '0.4rem',
              fontSize: '0.75rem',
            }}
          >
            CONFIDENCE
          </span>
        </div>
      </div>
    </div>
  );
}
