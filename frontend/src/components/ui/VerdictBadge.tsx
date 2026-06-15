'use client';

import { motion, useReducedMotion } from 'framer-motion';
import { ShieldAlert, ShieldCheck } from 'lucide-react';

interface VerdictBadgeProps {
  verdict: 'AI_GENERATED' | 'HUMAN';
}

export default function VerdictBadge({ verdict }: VerdictBadgeProps) {
  const shouldReduceMotion = useReducedMotion();

  const isAI = verdict === 'AI_GENERATED';
  const color = isAI ? '#F87171' : '#4ADE80';
  const bgColor = isAI ? 'rgba(248, 113, 113, 0.06)' : 'rgba(74, 222, 128, 0.06)';
  const label = isAI ? 'AI GENERATED' : 'HUMAN VERIFIED';
  const sublabel = isAI ? 'Synthetic voice signature detected' : 'Authentic voice signature confirmed';

  return (
    <motion.div
      initial={
        shouldReduceMotion
          ? { opacity: 1, y: 0 }
          : { opacity: 0, y: 12 }
      }
      animate={{ opacity: 1, y: 0 }}
      transition={
        shouldReduceMotion
          ? { duration: 0 }
          : { duration: 0.5, ease: 'easeOut' }
      }
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '0.75rem',
        padding: '1.25rem 2.5rem',
        borderRadius: '12px',
        background: bgColor,
        border: `1px solid ${color}`,
        position: 'relative',
        overflow: 'hidden',
        width: '100%',
        maxWidth: '360px',
      }}
    >
      {/* Subtle top accent line */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '2px',
          background: `linear-gradient(90deg, transparent, ${color}, transparent)`,
        }}
      />

      {/* Icon + Label row */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
        }}
      >
        {/* Icon */}
        <div
          style={{
            width: '36px',
            height: '36px',
            borderRadius: '8px',
            background: isAI ? 'rgba(248, 113, 113, 0.12)' : 'rgba(74, 222, 128, 0.12)',
            border: `1px solid ${isAI ? 'rgba(248, 113, 113, 0.25)' : 'rgba(74, 222, 128, 0.25)'}`,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color,
            flexShrink: 0,
          }}
        >
          {isAI ? <ShieldAlert size={18} /> : <ShieldCheck size={18} />}
        </div>

        <span
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '1.25rem',
            fontWeight: 700,
            color,
            letterSpacing: '0.08em',
          }}
        >
          {label}
        </span>
      </div>

      {/* Sublabel */}
      <span
        style={{
          fontFamily: 'var(--font-body)',
          fontSize: '0.8rem',
          color: 'var(--text-secondary)',
          letterSpacing: '0.02em',
        }}
      >
        {sublabel}
      </span>
    </motion.div>
  );
}
