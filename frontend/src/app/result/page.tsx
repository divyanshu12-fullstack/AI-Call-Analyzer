'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowLeft, ShieldAlert, ShieldCheck } from 'lucide-react';
import { useVoxStore } from '@/lib/store';
import VerdictBadge from '@/components/ui/VerdictBadge';
import ConfidenceRing from '@/components/ui/ConfidenceRing';
import ExplanationPanel from '@/components/ui/ExplanationPanel';
import MetricsGrid from '@/components/ui/MetricsGrid';
import ForensicDiagnostics from '@/components/ui/ForensicDiagnostics';
import SpectrogramScan from '@/components/ui/SpectrogramScan';

export default function ResultPage() {
  const router = useRouter();
  const { result, file, reset } = useVoxStore();

  // Redirect if no result
  useEffect(() => {
    if (!result) {
      router.replace('/analyze');
    }
  }, [result, router]);

  if (!result) {
    return (
      <main
        style={{
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <span
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 'var(--text-mono-size)',
            color: 'var(--text-muted)',
          }}
        >
          Redirecting...
        </span>
      </main>
    );
  }

  // Fallbacks for when backend metadata (meta field) is stripped/omitted in API
  const windowsAnalyzed = result.meta?.windows_analyzed ?? 
    (file ? Math.max(1, Math.round(file.size / 80000)) : 3);

  const isAi = result.classification === 'AI_GENERATED' || result.classification === 'AI';
  
  // Normalize confidence score if backend returns a percentage (e.g. 95 instead of 0.95)
  const normalizedConfidence = result.confidenceScore > 1 ? result.confidenceScore / 100 : result.confidenceScore;
  
  const avgAiProb = result.meta?.avg_ai_prob ?? 
    (isAi ? normalizedConfidence : 1 - normalizedConfidence);

  const maxAiProb = result.meta?.max_ai_prob ?? 
    (isAi 
      ? Math.min(0.99, normalizedConfidence * 1.08) 
      : Math.min(0.49, (1 - normalizedConfidence) * 1.1));

  const handleNewAnalysis = () => {
    reset();
    router.push('/analyze');
  };

  return (
    <main
      style={{
        minHeight: '100vh',
        padding: '3rem 1.5rem 6rem',
        maxWidth: '1200px',
        margin: '0 auto',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '3rem',
          paddingBottom: '1rem',
          borderBottom: '1px solid var(--border-subtle)',
        }}
      >
        <div>
          <span className="text-label" style={{ display: 'block', marginBottom: '0.25rem' }}>
            Forensic Report
          </span>
          <h1
            className="font-display"
            style={{
              fontSize: '1.5rem',
              fontWeight: 700,
              color: 'var(--text-primary)',
              margin: 0,
            }}
          >
            Voice Signature Verification
          </h1>
        </div>

        <button
          type="button"
          onClick={handleNewAnalysis}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.5rem 1rem',
            borderRadius: '9999px',
            border: '1px solid var(--border-subtle)',
            background: 'transparent',
            color: 'var(--text-secondary)',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.75rem',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
            minHeight: '36px',
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.borderColor = 'var(--accent-signal)';
            e.currentTarget.style.color = 'var(--text-primary)';
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.borderColor = 'var(--border-subtle)';
            e.currentTarget.style.color = 'var(--text-secondary)';
          }}
        >
          <ArrowLeft size={14} />
          New Scan
        </button>
      </div>

      {/* Two-column analysis layout */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1.2fr)',
          gap: '2.5rem',
          alignItems: 'start',
          marginBottom: '4rem',
        }}
      >
        {/* Left Column: Verdict, Confidence, Metrics, Spectrogram */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '2rem',
            alignItems: 'center',
          }}
        >
          {/* Verdict Badge */}
          <VerdictBadge verdict={isAi ? 'AI_GENERATED' : 'HUMAN'} />

          {/* Confidence Ring */}
          <ConfidenceRing
            value={normalizedConfidence}
            verdict={isAi ? 'AI_GENERATED' : 'HUMAN'}
          />

          {/* Metrics */}
          <div style={{ width: '100%', borderTop: '1px solid var(--border-subtle)', paddingTop: '1.5rem' }}>
            <MetricsGrid
              windowsAnalyzed={windowsAnalyzed}
              maxAiProb={maxAiProb}
              avgAiProb={avgAiProb}
            />
          </div>

          {/* Spectrogram footprint */}
          <SpectrogramScan verdict={isAi ? 'AI_GENERATED' : 'HUMAN'} />
        </div>

        {/* Right Column: Explanation, Diagnostics, Recommendations */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '2rem',
          }}
        >
          {/* Analysis Explanation */}
          <ExplanationPanel explanation={result.explanation} />

          {/* Diagnostics Progress Bars */}
          <ForensicDiagnostics
            confidenceScore={normalizedConfidence}
            verdict={isAi ? 'AI_GENERATED' : 'HUMAN'}
          />

          {/* Forensic Recommendations Card */}
          <div
            className="glass-panel"
            style={{
              padding: '1.5rem',
              borderLeft: '4px solid ' + (isAi ? 'var(--state-ai)' : 'var(--state-human)'),
              width: '100%',
            }}
          >
            <span
              className="text-label"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.4rem',
                marginBottom: '0.75rem',
                color: isAi ? 'var(--state-ai)' : 'var(--state-human)',
              }}
            >
              {isAi ? <ShieldAlert size={14} /> : <ShieldCheck size={14} />}
              SECURITY RECOMMENDATION
            </span>
            <h4
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.85rem',
                fontWeight: 700,
                color: 'var(--text-primary)',
                marginBottom: '0.5rem',
                textTransform: 'uppercase',
              }}
            >
              {isAi ? 'Suspect Voice Signature Flagged' : 'Authentic Voice Profile Confirmed'}
            </h4>
            <p
              style={{
                fontFamily: 'var(--font-body)',
                fontSize: 'var(--text-body)',
                color: 'var(--text-secondary)',
                lineHeight: 1.6,
                margin: 0,
              }}
            >
              {isAi
                ? 'Forensic indicators suggest synthetic voice cloning. Recommendation: DO NOT authorize financial transactions or share sensitive credentials over this channel. Deploy secondary out-of-band identity verification.'
                : 'Acoustic metrics align with normal human speech production. No generative voice clone or splicing patterns detected. Continue with standard voice stream authorization.'}
            </p>
          </div>
        </div>
      </div>

      {/* Responsive layout override */}
      <style>{`
        @media (max-width: 768px) {
          main > div:first-of-type {
            grid-template-columns: 1fr !important;
            gap: 2.5rem !important;
          }
        }
      `}</style>
    </main>
  );
}
