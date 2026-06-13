'use client';

interface MetricsGridProps {
  windowsAnalyzed: number;
  maxAiProb: number;
  avgAiProb: number;
}

interface MetricItemProps {
  value: string;
  label: string;
}

function MetricItem({ value, label }: MetricItemProps) {
  return (
    <div
      style={{
        borderTop: '1px solid var(--border-subtle)',
        paddingTop: '1.25rem',
        textAlign: 'center',
      }}
    >
      <div
        style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 'var(--text-title)',
          fontWeight: 700,
          color: 'var(--text-primary)',
          lineHeight: 1.2,
          marginBottom: '0.5rem',
        }}
      >
        {value}
      </div>
      <div
        className="text-label"
        style={{
          color: 'var(--text-muted)',
          fontSize: '0.65rem',
        }}
      >
        {label}
      </div>
    </div>
  );
}

export default function MetricsGrid({
  windowsAnalyzed,
  maxAiProb,
  avgAiProb,
}: MetricsGridProps) {
  return (
    <div style={{ width: '100%' }}>
      <span
        className="text-label"
        style={{ display: 'block', marginBottom: '1.25rem' }}
      >
        Inference Metrics
      </span>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: '1.5rem',
        }}
      >
        <MetricItem
          value={windowsAnalyzed.toString()}
          label="Windows Analyzed"
        />
        <MetricItem
          value={`${(maxAiProb * 100).toFixed(1)}%`}
          label="Max AI Probability"
        />
        <MetricItem
          value={`${(avgAiProb * 100).toFixed(1)}%`}
          label="Avg AI Probability"
        />
      </div>
    </div>
  );
}
