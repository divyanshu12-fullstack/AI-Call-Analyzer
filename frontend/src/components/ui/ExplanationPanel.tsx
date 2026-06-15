'use client';

import { useMemo, type ReactNode } from 'react';

interface ExplanationPanelProps {
  explanation: string;
}

// Technical terms to auto-highlight
const TECH_TERMS = [
  'synthetic spectral uniformity',
  'spectral uniformity',
  'MFCCs',
  'MFCC',
  'mel spectrogram',
  'Mel Spectrogram',
  'spectral contrast',
  'Spectral Contrast',
  'chroma STFT',
  'Chroma STFT',
  'zero crossing rate',
  'ZCR',
  'F0',
  'pitch tracking',
  'SE-VoiceResNet',
  'ResNet',
  'sliding window',
  'temperature scaling',
  'calibrated confidence',
  'focal loss',
  'squeeze-and-excitation',
  'attention mechanism',
  'spectral features',
  'acoustic channels',
  'synthetic speech',
  'deepfake',
  'AI-generated',
  'human speech',
  'spectral bandwidth',
  'spectral centroid',
];

function highlightTerms(text: string): ReactNode[] {
  // Sort terms by length descending to match longest first
  const sortedTerms = [...TECH_TERMS].sort((a, b) => b.length - a.length);
  const pattern = new RegExp(
    `(${sortedTerms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`,
    'gi'
  );

  const parts = text.split(pattern);
  return parts.map((part, i) => {
    const isMatch = sortedTerms.some(
      (term) => term.toLowerCase() === part.toLowerCase()
    );
    if (isMatch) {
      return (
        <mark key={i} className="tech-term">
          {part}
        </mark>
      );
    }
    return part;
  });
}

function InsightPoint({ children, index }: { children: ReactNode; index: number }) {
  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'flex-start',
        gap: '0.75rem',
        padding: '0.75rem',
        background: 'rgba(255, 255, 255, 0.02)',
        border: '1px solid var(--border-subtle)',
        borderRadius: '6px',
        marginBottom: '0.5rem',
      }}
    >
      <div
        style={{
          minWidth: '24px',
          height: '24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: 'rgba(56, 189, 248, 0.1)',
          color: 'var(--accent-signal)',
          borderRadius: '4px',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.7rem',
          fontWeight: 700,
        }}
      >
        {String(index).padStart(2, '0')}
      </div>
      <p
        style={{
          fontFamily: 'var(--font-body)',
          fontSize: '0.95rem',
          color: 'var(--text-secondary)',
          lineHeight: 1.6,
          margin: 0,
        }}
      >
        {children}
      </p>
    </div>
  );
}

export default function ExplanationPanel({ explanation }: ExplanationPanelProps) {
  // Split explanation into sentences for a structured breakdown
  const sentences = useMemo(() => {
    return explanation
      .split(/(?<=\.)\s+/)
      .filter((s) => s.trim().length > 10);
  }, [explanation]);

  return (
    <div
      className="glass-panel"
      style={{ padding: '1.5rem', width: '100%', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}
    >
      <div>
        <span
          className="text-label"
          style={{ display: 'block', marginBottom: '1rem' }}
        >
          Forensic Rationale Breakdown
        </span>

        <div>
          {sentences.length > 0 ? (
            sentences.map((sentence, idx) => (
              <InsightPoint key={idx} index={idx + 1}>
                {highlightTerms(sentence)}
              </InsightPoint>
            ))
          ) : (
            <InsightPoint index={1}>
              {highlightTerms(explanation)}
            </InsightPoint>
          )}
        </div>
      </div>

      <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '1.5rem' }}>
        <span
          className="text-label"
          style={{ display: 'block', marginBottom: '0.75rem', color: 'var(--text-muted)' }}
        >
          Neural Network Inference Trace
        </span>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '0.75rem',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.8rem',
            color: 'var(--text-secondary)',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem', background: 'rgba(0,0,0,0.2)', borderRadius: '4px' }}>
            <span>Architecture:</span>
            <span style={{ color: 'var(--accent-wave)' }}>SE-VoiceResNet-18</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem', background: 'rgba(0,0,0,0.2)', borderRadius: '4px' }}>
            <span>Parameters:</span>
            <span style={{ color: 'var(--accent-wave)' }}>11.4M (Quantized)</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem', background: 'rgba(0,0,0,0.2)', borderRadius: '4px' }}>
            <span>Input Tensor:</span>
            <span style={{ color: 'var(--accent-wave)' }}>[113, 256, 1]</span>
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.5rem', background: 'rgba(0,0,0,0.2)', borderRadius: '4px' }}>
            <span>Loss Criterion:</span>
            <span style={{ color: 'var(--accent-wave)' }}>Focal Loss (γ=2.0)</span>
          </div>
        </div>
      </div>
    </div>
  );
}
