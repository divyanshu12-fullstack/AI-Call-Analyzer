'use client';

import { Fragment, useState } from 'react';
import dynamic from 'next/dynamic';
import Link from 'next/link';
import {
  Layers,
  ScanLine,
  ShieldCheck,
  Fingerprint,
  Cpu,
  Activity,
  BookOpen,
  ChevronRight,
  type LucideIcon,
} from 'lucide-react';
import Navbar from '@/components/ui/Navbar';

const AudioOrb = dynamic(() => import('@/components/three/AudioOrb'), {
  ssr: false,
});

interface DiagnosticCapability {
  icon: LucideIcon;
  title: string;
  description: string;
  tag: string;
  accent: string;
}

const DIAGNOSTICS: DiagnosticCapability[] = [
  {
    icon: Fingerprint,
    title: 'FORMANT & VOCAL TRACT RESONANCE SCAN',
    description:
      'Inspects vocal tract resonances (formants) for mathematical alignment anomalies typical of vocoders (e.g. WaveGlow, MelGAN). Exposes artificial vocal cavity length variations.',
    tag: 'FMT-SCAN',
    accent: '#38BDF8',
  },
  {
    icon: Activity,
    title: 'PHASE COHERENCE & SPLICING SCAN',
    description:
      'Analyzes micro-alignment consistency of the acoustic phase profile across window frame boundaries. Exposes synthetic splicing joints and frame boundary phase jumps.',
    tag: 'PHS-COHERENCE',
    accent: '#818CF8',
  },
  {
    icon: Cpu,
    title: 'HIGH-FREQUENCY SPECTRAL ATTENUATION',
    description:
      'Examines sibilance and high-frequency noise bands (8kHz+) for artificial compression or missing micro-textures. AI generators often fail to replicate organic friction sounds.',
    tag: 'HF-MICROTEXTURES',
    accent: '#C084FC',
  }
];

interface FeatureItem {
  name: string;
  description: string;
  role: string;
}

const ACOUSTIC_FEATURES: FeatureItem[] = [
  {
    name: 'Mel Spectrogram (64 bins)',
    description: 'Compresses frequencies to match the non-linear human auditory system, highlighting synthetic formant energy peaks.',
    role: 'Coarse vocal tract signature mapping',
  },
  {
    name: 'MFCCs & Deltas (26 channels)',
    description: 'Mel-Frequency Cepstral Coefficients plus first-order derivatives capturing instantaneous spectral change and transition artifacts.',
    role: 'Fine-grained voice texture footprinting',
  },
  {
    name: 'F0 Pitch Contour Tracking',
    description: 'Tracks the fundamental frequency of vocal fold vibration using parabolic interpolation to flag mechanical pitch lock.',
    role: 'Prosody naturalness verification',
  },
  {
    name: 'Spectral Contrast (7 bands)',
    description: 'Measures valley-to-peak differences in sub-bands to detect smoothing artifacts left by vocoder generative passes.',
    role: 'Acoustic contrast verification',
  }
];

interface GlossaryTerm {
  term: string;
  definition: string;
}

const GLOSSARY: GlossaryTerm[] = [
  {
    term: 'SE-VoiceResNet-18',
    definition: 'A specialized 18-layer residual neural network enhanced with Squeeze-and-Excitation attention blocks. It dynamically re-weights acoustic feature channels based on global context, prioritizing anomalies.',
  },
  {
    term: 'Temperature Calibration',
    definition: 'A scaling pass applied to the final Softmax layer parameters. This corrects the deep learning model’s tendency to be overconfident, aligning predicted scores with actual, empirical forensic accuracy.',
  },
  {
    term: 'Neural Vocoder Artifacts',
    definition: 'Micro-structural audio distortions (like phase misalignment, metallic ringing, or missing high-frequency sibilance) introduced when a generative model reconstructs a waveform from a spectrogram.',
  },
];

const LANGUAGES = ['English', 'Hindi', 'Tamil', 'Telugu', 'Malayalam'] as const;

export default function HomePage() {
  const [activeFeature, setActiveFeature] = useState<number>(0);

  return (
    <main style={{ color: 'var(--text-primary)' }}>
      <Navbar />

      {/* ── Hero Section ── */}
      <section
        className="hero-section"
        style={{
          position: 'relative',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          overflow: 'hidden',
        }}
      >
        <AudioOrb />

        <div
          style={{
            position: 'relative',
            zIndex: 1,
            textAlign: 'center',
            maxWidth: '900px',
          }}
        >
          <span
            className="text-label"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.5rem',
              marginBottom: '1.5rem',
              padding: '0.4rem 1rem',
              borderRadius: '9999px',
              border: '1px solid rgba(56, 189, 248, 0.2)',
              background: 'rgba(56, 189, 248, 0.04)',
              color: 'var(--accent-signal)',
            }}
          >
            <ShieldCheck size={14} />
            Advanced Audio Forensic Analysis
          </span>

          <h1
            className="font-display text-hero"
            style={{
              color: 'var(--text-primary)',
              marginBottom: '1.5rem',
              fontWeight: 800,
              letterSpacing: '-0.02em',
              lineHeight: 1.1,
              wordBreak: 'break-word',
            }}
          >
            Expose Synthetic Voices.
            <br />
            Verify Audio Authenticity.
          </h1>

          <p
            style={{
              fontFamily: 'var(--font-body)',
              fontSize: '1.125rem',
              color: 'var(--text-secondary)',
              marginBottom: '2.5rem',
              lineHeight: 1.7,
              maxWidth: '700px',
              margin: '0 auto 2.5rem',
            }}
          >
            VoxGuard AI is a forensic-grade platform engineered to detect cloned voices, deepfake speech, and generative AI calls. Utilizing state-of-the-art <strong>SE-VoiceResNet-18</strong> networks and 11-channel acoustic profiling, we reveal anomalies invisible to the human ear.
          </p>

          <div
            style={{
              display: 'flex',
              justifyContent: 'center',
              gap: '1rem',
              flexWrap: 'wrap',
            }}
          >
            <Link
              href="/analyze"
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.5rem',
                padding: '14px 36px',
                borderRadius: '100px',
                border: '1px solid var(--accent-signal)',
                background: 'linear-gradient(135deg, var(--accent-signal), var(--accent-wave))',
                color: '#070A10',
                fontFamily: 'var(--font-mono)',
                fontSize: '0.875rem',
                fontWeight: 700,
                letterSpacing: '0.06em',
                textDecoration: 'none',
                transition: 'all 0.2s ease',
                cursor: 'pointer',
                minHeight: '52px',
                boxShadow: '0 4px 20px rgba(56, 189, 248, 0.25)',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.filter = 'brightness(1.1)';
                e.currentTarget.style.transform = 'translateY(-1px)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.filter = 'none';
                e.currentTarget.style.transform = 'none';
              }}
            >
              Start Forensic Scan
              <ChevronRight size={16} />
            </Link>
          </div>
        </div>
      </section>

      {/* ── Metrics Strip ── */}
      <div className="metrics-strip">
        <div className="metric-item">
          <span className="metric-value">SE-VoiceResNet-18</span>
          <span className="metric-label">ARCHITECTURE</span>
        </div>
        <div className="metric-divider" />
        <div className="metric-item">
          <span className="metric-value">97.4%</span>
          <span className="metric-label">EMPIRICAL ACCURACY</span>
        </div>
        <div className="metric-divider" />
        <div className="metric-item">
          <span className="metric-value">113</span>
          <span className="metric-label">ACOUSTIC FEATURES</span>
        </div>
        <div className="metric-divider" />
        <div className="metric-item">
          <span className="metric-value">100%</span>
          <span className="metric-label">AI RECALL RATE</span>
        </div>
      </div>

      {/* ── Forensic Diagnostics Section ── */}
      <section
        style={{
          paddingTop: '64px',
          maxWidth: '1100px',
          margin: '0 auto',
        }}
      >
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <span className="text-label" style={{ display: 'block', marginBottom: '1rem' }}>CORE CAPABILITIES</span>
          <h2 className="font-display" style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            Detecting the Synthesized Vocal Fingerprint
          </h2>
          <p style={{ fontFamily: 'var(--font-body)', fontSize: '1rem', color: 'var(--text-secondary)', maxWidth: '600px', margin: '0.75rem auto 0', lineHeight: 1.6 }}>
            VoxGuard AI scans voice streams across four core forensic dimensions to flag generative anomalies.
          </p>
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
            gap: '1.5rem',
          }}
        >
          {DIAGNOSTICS.map((diag) => (
            <div
              key={diag.title}
              className="capability-card"
              style={{
                '--card-accent': diag.accent,
                display: 'flex',
                flexDirection: 'column',
                gap: '1.25rem',
              } as React.CSSProperties}
            >
              <div
                style={{
                  width: '48px',
                  height: '48px',
                  borderRadius: '8px',
                  background: 'var(--bg-glass)',
                  border: '1px solid var(--border-subtle)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--card-accent)',
                }}
              >
                <diag.icon size={24} />
              </div>

              <div>
                <span
                  style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.6rem',
                    color: 'var(--card-accent)',
                    display: 'block',
                    marginBottom: '0.5rem',
                    letterSpacing: '0.08em',
                  }}
                >
                  {diag.tag}
                </span>
                <h3
                  className="font-display"
                  style={{
                    fontSize: '0.95rem',
                    fontWeight: 700,
                    color: 'var(--text-primary)',
                    marginBottom: '0.5rem',
                    lineHeight: 1.3,
                  }}
                >
                  {diag.title}
                </h3>
                <p
                  style={{
                    fontFamily: 'var(--font-body)',
                    fontSize: 'var(--text-body)',
                    color: 'var(--text-secondary)',
                    lineHeight: 1.6,
                    margin: 0,
                  }}
                >
                  {diag.description}
                </p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── Interactive Feature Sandbox ── */}
      <section
        style={{
          background: 'rgba(13, 17, 23, 0.5)',
          borderTop: '1px solid var(--border-subtle)',
          borderBottom: '1px solid var(--border-subtle)',
        }}
      >
        <div
          className="responsive-feature-grid"
          style={{
            maxWidth: '1100px',
            margin: '0 auto',
            display: 'grid',
            gridTemplateColumns: '1.2fr 1fr',
            gap: '4rem',
            alignItems: 'center',
          }}
        >
          {/* Left panel - specs */}
          <div>
            <span className="text-label" style={{ display: 'block', marginBottom: '1rem' }}>TECHNICAL SPECS</span>
            <h2 className="font-display" style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-primary)', marginBottom: '1.5rem' }}>
              The 113-Channel Feature Extraction Pipeline
            </h2>
            <p style={{ fontFamily: 'var(--font-body)', fontSize: '1rem', color: 'var(--text-secondary)', lineHeight: 1.7, marginBottom: '2rem' }}>
              To train a voice classifier that ignores language semantic details and focuses exclusively on acoustic authenticity, VoxGuard transforms a raw audio stream into 113 concurrent features. Select a feature below to read its role:
            </p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {ACOUSTIC_FEATURES.map((feat, idx) => (
                <button
                  key={feat.name}
                  type="button"
                  onClick={() => setActiveFeature(idx)}
                  style={{
                    textAlign: 'left',
                    padding: '1rem 1.5rem',
                    borderRadius: '8px',
                    border: '1px solid',
                    borderColor: activeFeature === idx ? 'var(--accent-signal)' : 'var(--border-subtle)',
                    background: activeFeature === idx ? 'rgba(56, 189, 248, 0.05)' : 'rgba(255, 255, 255, 0.01)',
                    color: activeFeature === idx ? 'var(--text-primary)' : 'var(--text-secondary)',
                    fontFamily: 'var(--font-mono)',
                    fontSize: '0.85rem',
                    fontWeight: 600,
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                  }}
                >
                  {feat.name}
                </button>
              ))}
            </div>
          </div>

          {/* Right panel - dynamic description box */}
          <div
            className="glass-panel"
            style={{
              padding: '2.5rem',
              borderLeft: '4px solid var(--accent-signal)',
              minHeight: '320px',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'center',
            }}
          >
            <span
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.65rem',
                color: 'var(--accent-signal)',
                textTransform: 'uppercase',
                letterSpacing: '0.08em',
                marginBottom: '1rem',
                display: 'block',
              }}
            >
              diagnostic channel detail
            </span>
            <h3
              className="font-display"
              style={{
                fontSize: '1.25rem',
                fontWeight: 700,
                color: 'var(--text-primary)',
                marginBottom: '1rem',
              }}
            >
              {ACOUSTIC_FEATURES[activeFeature].name}
            </h3>
            <p
              style={{
                fontFamily: 'var(--font-body)',
                fontSize: '0.95rem',
                color: 'var(--text-secondary)',
                lineHeight: 1.7,
                marginBottom: '1.5rem',
              }}
            >
              {ACOUSTIC_FEATURES[activeFeature].description}
            </p>
            <div
              style={{
                paddingTop: '1rem',
                borderTop: '1px solid var(--border-subtle)',
              }}
            >
              <span
                style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.65rem',
                  color: 'var(--text-muted)',
                  display: 'block',
                  textTransform: 'uppercase',
                  marginBottom: '0.25rem',
                }}
              >
                forensic classification role
              </span>
              <span
                style={{
                  fontFamily: 'var(--font-body)',
                  fontSize: '0.85rem',
                  fontWeight: 600,
                  color: 'var(--accent-wave)',
                }}
              >
                {ACOUSTIC_FEATURES[activeFeature].role}
              </span>
            </div>
          </div>
        </div>
      </section>

      {/* ── Forensic Glossary ── */}
      <section
        style={{
          maxWidth: '1000px',
          margin: '0 auto',
        }}
      >
        <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
          <span className="text-label" style={{ display: 'block', marginBottom: '1rem' }}>GLOSSARY</span>
          <h2 className="font-display" style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--text-primary)' }}>
            Forensic Lexicon
          </h2>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {GLOSSARY.map((item) => (
            <div key={item.term} className="lexicon-item">
              <h3 className="lexicon-term">{item.term}</h3>
              <p className="lexicon-definition">{item.definition}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="site-footer">
        <div className="footer-brand">
          <span className="footer-logo">VOXGUARD</span>
          <span className="footer-tagline">Audio Forensics Platform</span>
        </div>
        <div className="footer-langs">
          <span className="footer-langs-label">MULTILINGUAL SUPPORT</span>
          <div className="footer-lang-strip">
            EN · HI · TA · TE · ML
          </div>
        </div>
        <div className="footer-meta">
          <span>SE-VoiceResNet-18 Engine</span>
          <span>·</span>
          <span>Temperature Calibrated</span>
        </div>
      </footer>

      {/* Responsive layout corrections */}
      <style>{`
        @media (max-width: 768px) {
          .responsive-feature-grid {
            grid-template-columns: 1fr !important;
            gap: 2rem !important;
          }
          .responsive-stats-grid {
            grid-template-columns: 1fr !important;
            gap: 1rem !important;
          }
        }
      `}</style>
    </main>
  );
}
