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
  AudioWaveform,
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
            padding: '2rem 1.5rem',
            borderRadius: '24px',
            background: 'radial-gradient(ellipse 80% 80% at 50% 50%, rgba(7,10,16,0.65) 0%, rgba(7,10,16,0.3) 60%, transparent 100%)',
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
              textShadow: '0 2px 20px rgba(0,0,0,0.7), 0 0 40px rgba(0,0,0,0.5)',
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
              color: 'rgba(203, 213, 225, 0.95)',
              marginBottom: '2.5rem',
              lineHeight: 1.7,
              maxWidth: '700px',
              margin: '0 auto 2.5rem',
              textShadow: '0 1px 12px rgba(0,0,0,0.8), 0 0 30px rgba(0,0,0,0.5)',
            }}
          >
            VoxGuard AI is a forensic-grade platform engineered to detect cloned voices, deepfake speech, and generative AI calls. Utilizing state-of-the-art <strong style={{ color: 'var(--accent-signal)' }}>SE-VoiceResNet-18</strong> networks and 11-channel acoustic profiling, we reveal anomalies invisible to the human ear.
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
              className="hero-cta-btn"
            >
              <span className="hero-cta-shimmer" />
              <AudioWaveform size={18} strokeWidth={2.2} />
              <span>Start Forensic Scan</span>
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
        </div>
      </footer>

      {/* Responsive layout corrections */}
      <style>{`
        /* ── Hero CTA Button ── */
        .hero-cta-btn {
          position: relative;
          display: inline-flex;
          align-items: center;
          gap: 0.6rem;
          padding: 16px 40px;
          border-radius: 100px;
          border: none;
          background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
          color: #070A10;
          font-family: var(--font-mono);
          font-size: 0.875rem;
          font-weight: 700;
          letter-spacing: 0.06em;
          text-decoration: none;
          cursor: pointer;
          min-height: 54px;
          overflow: hidden;
          isolation: isolate;
          transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.3s ease;
          box-shadow:
            0 0 20px rgba(56, 189, 248, 0.3),
            0 0 60px rgba(56, 189, 248, 0.1),
            0 4px 16px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
          animation: hero-cta-pulse 3s ease-in-out infinite;
        }

        .hero-cta-btn:hover {
          transform: translateY(-2px) scale(1.03);
          box-shadow:
            0 0 30px rgba(56, 189, 248, 0.45),
            0 0 80px rgba(129, 140, 248, 0.2),
            0 8px 32px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.25);
        }

        .hero-cta-btn:active {
          transform: translateY(0px) scale(0.98);
        }

        /* Shimmer sweep effect */
        .hero-cta-shimmer {
          position: absolute;
          inset: 0;
          background: linear-gradient(
            105deg,
            transparent 35%,
            rgba(255, 255, 255, 0.25) 45%,
            rgba(255, 255, 255, 0.35) 50%,
            rgba(255, 255, 255, 0.25) 55%,
            transparent 65%
          );
          transform: translateX(-100%);
          animation: hero-shimmer 4s ease-in-out infinite;
          z-index: -1;
        }

        @keyframes hero-shimmer {
          0%, 100% { transform: translateX(-100%); }
          50% { transform: translateX(100%); }
        }

        @keyframes hero-cta-pulse {
          0%, 100% { box-shadow: 0 0 20px rgba(56, 189, 248, 0.3), 0 0 60px rgba(56, 189, 248, 0.1), 0 4px 16px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.2); }
          50% { box-shadow: 0 0 28px rgba(56, 189, 248, 0.4), 0 0 80px rgba(129, 140, 248, 0.15), 0 4px 16px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.2); }
        }

        @media (max-width: 768px) {
          .responsive-feature-grid {
            grid-template-columns: 1fr !important;
            gap: 2rem !important;
            padding: 0 0.5rem !important;
          }
          .responsive-stats-grid {
            grid-template-columns: 1fr !important;
            gap: 1rem !important;
          }
          .text-hero {
            font-size: clamp(2rem, 8vw, 3.5rem) !important;
          }
          .hero-cta-btn {
            padding: 14px 32px;
            font-size: 0.8rem;
            min-height: 50px;
            gap: 0.5rem;
          }
        }
        @media (max-width: 480px) {
          .text-hero {
            font-size: clamp(1.75rem, 7vw, 2.5rem) !important;
          }
          .hero-cta-btn {
            padding: 13px 28px;
            font-size: 0.75rem;
            min-height: 46px;
          }
        }
      `}</style>
    </main>
  );
}
