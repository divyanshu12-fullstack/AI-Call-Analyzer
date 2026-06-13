'use client';

import { motion, useReducedMotion } from 'framer-motion';

interface VerdictBadgeProps {
  verdict: 'AI_GENERATED' | 'HUMAN';
}

export default function VerdictBadge({ verdict }: VerdictBadgeProps) {
  const shouldReduceMotion = useReducedMotion();

  const isAI = verdict === 'AI_GENERATED';
  const color = isAI ? 'var(--state-ai)' : 'var(--state-human)';
  const bgColor = isAI ? 'rgba(248, 113, 113, 0.1)' : 'rgba(74, 222, 128, 0.1)';
  const label = isAI ? 'AI GENERATED' : 'HUMAN';

  return (
    <motion.div
      initial={
        shouldReduceMotion
          ? { opacity: 1, scale: 1 }
          : { opacity: 0, scale: 0.9 }
      }
      animate={{ opacity: 1, scale: 1 }}
      transition={
        shouldReduceMotion
          ? { duration: 0 }
          : { duration: 0.4, ease: 'easeOut' }
      }
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '0.75rem',
        padding: '0.75rem 2rem',
        borderRadius: '9999px',
        background: bgColor,
        border: `1px solid ${color}`,
      }}
    >
      {/* Indicator dot */}
      <span
        style={{
          width: '10px',
          height: '10px',
          borderRadius: '50%',
          background: color,
          boxShadow: `0 0 8px ${color}`,
        }}
      />

      <span
        style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 'var(--text-title)',
          fontWeight: 700,
          color,
          letterSpacing: '0.06em',
        }}
      >
        {label}
      </span>
    </motion.div>
  );
}
