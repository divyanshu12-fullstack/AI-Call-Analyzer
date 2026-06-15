'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import dynamic from 'next/dynamic';
import { ArrowLeft, Terminal, Activity } from 'lucide-react';
import Link from 'next/link';
import { useVoxStore } from '@/lib/store';
import { analyzeAudio } from '@/lib/api';
import { getAudioFormat } from '@/lib/audioUtils';
import FileDropzone from '@/components/ui/FileDropzone';
import LanguageSelector from '@/components/ui/LanguageSelector';

const WaveformScan = dynamic(
  () => import('@/components/three/WaveformScan'),
  { ssr: false }
);

const ParticlePreview = dynamic(
  () => import('@/components/three/ParticlePreview'),
  { ssr: false }
);

function EmptyPreview() {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100%',
        minHeight: '300px',
        gap: '1.5rem',
      }}
    >
      <div style={{ width: '100%', height: '160px', position: 'relative', opacity: 0.5 }}>
        <ParticlePreview />
      </div>

      <span
        style={{
          fontFamily: 'var(--font-mono)',
          fontSize: 'var(--text-mono-size)',
          color: 'var(--text-muted)',
        }}
      >
        Drop a file to see a preview
      </span>
    </div>
  );
}

export default function AnalyzePage() {
  const router = useRouter();
  const {
    file,
    language,
    isAnalyzing,
    error,
    setIsAnalyzing,
    setResult,
    setError,
  } = useVoxStore();

  const [scanLogs, setScanLogs] = useState<string[]>([]);

  const handleAnalyze = async () => {
    if (!file) return;

    setIsAnalyzing(true);
    setScanLogs(['[SYS] Initializing SE-VoiceResNet-18 pipeline...']);
    
    // Simulate streaming logs for UX realism
    const logs = [
      '[EXT] Extracting 113-channel acoustic features...',
      '[DSP] Computing Mel-frequency cepstral coefficients (MFCCs)...',
      '[DSP] Running Phase Coherence cross-correlation...',
      '[SCAN] Isolating high-frequency micro-textures (8kHz+)...',
      '[SCAN] Analyzing prosody and pitch variations...',
      '[NN] Forward pass through SE-VoiceResNet-18...',
      '[NN] Applying focal loss temperature calibration...',
      '[SYS] Compiling forensic report...'
    ];

    let logIndex = 0;
    const logInterval = setInterval(() => {
      if (logIndex < logs.length) {
        setScanLogs(prev => [...prev, logs[logIndex]]);
        logIndex++;
      }
    }, 400); // Add a new log every 400ms

    try {
      const format = getAudioFormat(file);
      const result = await analyzeAudio(file, language, format);
      
      clearInterval(logInterval);
      setResult(result);
      router.push('/result');
    } catch (err: unknown) {
      clearInterval(logInterval);
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('Analysis failed. Please try again.');
      }
      setIsAnalyzing(false);
    }
  };

  // Loading state
  if (isAnalyzing) {
    return (
      <main
        style={{
          minHeight: '100vh',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          padding: '2rem',
          background: 'var(--bg-void)',
        }}
      >
        <div style={{ width: '100%', maxWidth: '600px', display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <WaveformScan />
          </div>
          
          <div
            className="glass-panel"
            style={{
              padding: '1.5rem',
              background: '#0D1117',
              border: '1px solid var(--border-subtle)',
              borderRadius: '8px',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.8rem',
              color: 'var(--accent-signal)',
              height: '250px',
              overflowY: 'auto',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.5rem',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem', color: 'var(--text-muted)' }}>
              <Terminal size={14} />
              <span>FORENSIC TERMINAL OUTPUT</span>
            </div>
            {scanLogs.map((log, i) => (
              <div key={i} style={{ opacity: 0.8, animation: 'fadeIn 0.3s ease-in-out' }}>
                <span style={{ color: 'var(--text-muted)', marginRight: '0.5rem' }}>{new Date().toISOString().substring(11, 23)}</span>
                <span style={{ color: log?.includes('[SYS]') ? 'var(--text-primary)' : log?.includes('[SCAN]') ? 'var(--state-uncertain)' : 'var(--accent-wave)' }}>{log}</span>
              </div>
            ))}
            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', marginTop: 'auto', opacity: 0.6 }}>
              <Activity size={14} className="animate-pulse" />
              <span>Processing data stream...</span>
            </div>
          </div>
        </div>
        <style>{`
          @keyframes fadeIn {
            from { opacity: 0; transform: translateY(5px); }
            to { opacity: 0.8; transform: translateY(0); }
          }
        `}</style>
      </main>
    );
  }

  return (
    <main
      style={{
        minHeight: '100vh',
        padding: '2rem 1.5rem',
        maxWidth: '1200px',
        margin: '0 auto',
      }}
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '1rem',
          marginBottom: '3rem',
          paddingTop: '1rem',
        }}
      >
        <Link
          href="/"
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: '44px',
            height: '44px',
            borderRadius: '50%',
            border: '1px solid var(--border-subtle)',
            color: 'var(--text-secondary)',
            textDecoration: 'none',
            transition: 'border-color 0.2s ease',
          }}
          aria-label="Back to home"
        >
          <ArrowLeft size={18} />
        </Link>
        <div>
          <span className="text-label" style={{ display: 'block', marginBottom: '0.25rem' }}>
            Analyzer Console
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
            Audio Analysis
          </h1>
        </div>
      </div>

      {/* Two-column layout */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(0, 1fr) minmax(0, 1fr)',
          gap: '2rem',
          alignItems: 'start',
        }}
      >
        {/* Left panel - Input */}
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '2rem',
          }}
        >
          <LanguageSelector />
          <FileDropzone />

          {/* Error display */}
          {error && (
            <div
              style={{
                padding: '1rem',
                borderRadius: '8px',
                border: '1px solid var(--state-ai)',
                background: 'rgba(248, 113, 113, 0.05)',
              }}
            >
              <span
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: 'var(--text-mono-size)',
                  color: 'var(--state-ai)',
                }}
              >
                {error}
              </span>
            </div>
          )}

          {/* Analyze button */}
          <button
            type="button"
            onClick={handleAnalyze}
            disabled={!file}
            style={{
              width: '100%',
              padding: '1rem',
              borderRadius: '12px',
              border: 'none',
              background: file
                ? 'linear-gradient(135deg, var(--accent-signal), var(--accent-wave))'
                : 'linear-gradient(135deg, var(--accent-signal), var(--accent-wave))',
              color: '#070A10',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.875rem',
              fontWeight: 700,
              letterSpacing: '0.06em',
              cursor: file ? 'pointer' : 'not-allowed',
              opacity: file ? 1 : 0.4,
              transition: 'all 0.2s ease',
              minHeight: '52px',
            }}
            onMouseEnter={(e) => {
              if (file) {
                e.currentTarget.style.filter = 'brightness(1.1)';
                e.currentTarget.style.transform = 'translateY(-1px)';
              }
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.filter = 'none';
              e.currentTarget.style.transform = 'none';
            }}
          >
            Run Analysis
          </button>
        </div>

        {/* Right panel - Preview */}
        <div
          className="glass-panel"
          style={{
            padding: '1.5rem',
            minHeight: '400px',
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          <span
            className="text-label"
            style={{ display: 'block', marginBottom: '1rem' }}
          >
            Audio Preview
          </span>

          {file ? (
            <div style={{ flex: 1 }}>
              {/* Waveform is shown inside FileDropzone when file is loaded */}
              <div
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: '100%',
                  minHeight: '250px',
                  gap: '1rem',
                }}
              >
                <div style={{ width: '100%', height: '140px', position: 'relative' }}>
                  <ParticlePreview />
                </div>

                <div
                  style={{
                    textAlign: 'center',
                    fontFamily: 'var(--font-mono)',
                    fontSize: 'var(--text-mono-size)',
                    color: 'var(--text-secondary)',
                  }}
                >
                  <div>{file.name}</div>
                  <div style={{ color: 'var(--text-muted)', marginTop: '0.25rem', marginBottom: '1.5rem' }}>
                    Ready for analysis
                  </div>
                </div>

                <div
                  style={{
                    width: '100%',
                    padding: '1.25rem',
                    borderRadius: '8px',
                    background: 'rgba(255, 255, 255, 0.01)',
                    border: '1px solid var(--border-subtle)',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.7rem',
                    color: 'var(--text-secondary)',
                    lineHeight: 1.6,
                    textAlign: 'left',
                  }}
                >
                  <div
                    style={{
                      color: 'var(--accent-signal)',
                      fontWeight: 700,
                      marginBottom: '0.5rem',
                      textTransform: 'uppercase',
                      letterSpacing: '0.05em',
                    }}
                  >
                    Forensic Pipeline Preparation
                  </div>
                  • Decomposing signal into 113 unique acoustic feature channels<br />
                  • Segmenting track into 5s overlapping analysis windows<br />
                  • Normalizing power scale via Mel frequency weighting
                </div>
              </div>
            </div>
          ) : (
            <EmptyPreview />
          )}
        </div>
      </div>

      {/* Responsive style override */}
      <style>{`
        @media (max-width: 768px) {
          main > div:last-of-type {
            grid-template-columns: 1fr !important;
          }
        }
      `}</style>
    </main>
  );
}
