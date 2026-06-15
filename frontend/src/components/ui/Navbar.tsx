'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { AudioWaveform } from 'lucide-react';

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <>
      <nav className={`vg-navbar${scrolled ? ' vg-navbar--scrolled' : ''}`}>
        {/* Gradient accent line */}
        <div className="vg-navbar__accent" />

        <div className="vg-navbar__inner">
          {/* Logo / Wordmark */}
          <Link href="/" className="vg-navbar__brand">
            <span className="vg-navbar__logo-text">VOXGUARD</span>
            <span className="vg-navbar__logo-badge">AI</span>
          </Link>

          {/* CTA Button */}
          <Link href="/analyze" className="vg-navbar__cta">
            <span className="vg-navbar__cta-glow" />
            <AudioWaveform size={15} strokeWidth={2.2} />
            <span>Analyze Audio</span>
          </Link>
        </div>
      </nav>

      <style>{`
        /* ── Navbar Container ── */
        .vg-navbar {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          z-index: 100;
          height: 64px;
          background: rgba(7, 10, 16, 0.6);
          backdrop-filter: blur(20px) saturate(1.4);
          -webkit-backdrop-filter: blur(20px) saturate(1.4);
          border-bottom: 1px solid rgba(255, 255, 255, 0.06);
          transition: background 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease;
        }

        .vg-navbar--scrolled {
          background: rgba(7, 10, 16, 0.92);
          border-bottom-color: rgba(56, 189, 248, 0.1);
          box-shadow:
            0 1px 0 0 rgba(56, 189, 248, 0.06),
            0 8px 32px rgba(0, 0, 0, 0.4);
        }

        /* ── Animated Gradient Accent Line ── */
        .vg-navbar__accent {
          position: absolute;
          bottom: -1px;
          left: 0;
          right: 0;
          height: 1px;
          background: linear-gradient(
            90deg,
            transparent 0%,
            rgba(56, 189, 248, 0.0) 10%,
            rgba(56, 189, 248, 0.5) 30%,
            rgba(129, 140, 248, 0.5) 50%,
            rgba(192, 132, 252, 0.5) 70%,
            rgba(56, 189, 248, 0.0) 90%,
            transparent 100%
          );
          background-size: 200% 100%;
          animation: vg-accent-sweep 6s ease-in-out infinite;
          opacity: 0.7;
        }

        .vg-navbar--scrolled .vg-navbar__accent {
          opacity: 1;
        }

        @keyframes vg-accent-sweep {
          0%, 100% { background-position: 200% 0; }
          50% { background-position: -200% 0; }
        }

        /* ── Inner Layout ── */
        .vg-navbar__inner {
          max-width: 1400px;
          margin: 0 auto;
          height: 100%;
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 0 2rem;
        }

        /* ── Brand / Logo ── */
        .vg-navbar__brand {
          display: flex;
          align-items: center;
          gap: 0.6rem;
          text-decoration: none;
          transition: opacity 0.2s ease;
        }

        .vg-navbar__brand:hover {
          opacity: 0.85;
        }

        .vg-navbar__logo-icon {
          display: flex;
          align-items: center;
          justify-content: center;
          width: 32px;
          height: 32px;
          border-radius: 8px;
          background: linear-gradient(135deg, rgba(56, 189, 248, 0.15), rgba(129, 140, 248, 0.1));
          border: 1px solid rgba(56, 189, 248, 0.2);
          color: var(--accent-signal, #38BDF8);
          flex-shrink: 0;
        }

        .vg-navbar__logo-text {
          font-family: var(--font-display, 'Cabinet Grotesk', sans-serif);
          font-weight: 800;
          font-size: 1.15rem;
          color: var(--text-primary, #F1F5F9);
          letter-spacing: 0.06em;
        }

        .vg-navbar__logo-badge {
          font-family: var(--font-mono, 'JetBrains Mono', monospace);
          font-size: 0.55rem;
          font-weight: 700;
          color: var(--accent-signal, #38BDF8);
          background: rgba(56, 189, 248, 0.1);
          border: 1px solid rgba(56, 189, 248, 0.2);
          border-radius: 4px;
          padding: 1px 5px;
          letter-spacing: 0.08em;
          line-height: 1.4;
          margin-top: -6px;
        }

        /* ── CTA Button ── */
        .vg-navbar__cta {
          position: relative;
          display: inline-flex;
          align-items: center;
          gap: 0.5rem;
          padding: 9px 22px;
          border-radius: 100px;
          border: 1px solid rgba(56, 189, 248, 0.35);
          background: linear-gradient(
            135deg,
            rgba(56, 189, 248, 0.12) 0%,
            rgba(129, 140, 248, 0.08) 100%
          );
          color: var(--text-primary, #F1F5F9);
          font-family: var(--font-mono, 'JetBrains Mono', monospace);
          font-size: 0.75rem;
          font-weight: 600;
          letter-spacing: 0.06em;
          text-decoration: none;
          cursor: pointer;
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          overflow: hidden;
          isolation: isolate;
        }

        .vg-navbar__cta-glow {
          position: absolute;
          inset: -1px;
          border-radius: inherit;
          background: linear-gradient(
            135deg,
            rgba(56, 189, 248, 0.2) 0%,
            rgba(129, 140, 248, 0.15) 50%,
            rgba(192, 132, 252, 0.1) 100%
          );
          opacity: 0;
          transition: opacity 0.3s ease;
          z-index: -1;
        }

        .vg-navbar__cta:hover {
          border-color: rgba(56, 189, 248, 0.6);
          transform: translateY(-1px);
          box-shadow:
            0 0 20px rgba(56, 189, 248, 0.2),
            0 0 40px rgba(56, 189, 248, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.06);
        }

        .vg-navbar__cta:hover .vg-navbar__cta-glow {
          opacity: 1;
        }

        .vg-navbar__cta:active {
          transform: translateY(0px) scale(0.98);
        }

        /* ── Mobile Responsive ── */
        @media (max-width: 768px) {
          .vg-navbar {
            height: 58px;
          }
          .vg-navbar__inner {
            padding: 0 1rem;
          }
          .vg-navbar__logo-icon {
            width: 28px;
            height: 28px;
            border-radius: 6px;
          }
          .vg-navbar__logo-icon svg {
            width: 14px;
            height: 14px;
          }
          .vg-navbar__logo-text {
            font-size: 1rem;
          }
          .vg-navbar__logo-badge {
            font-size: 0.5rem;
            padding: 0px 4px;
          }
          .vg-navbar__cta {
            padding: 8px 16px;
            font-size: 0.7rem;
            gap: 0.35rem;
          }
          .vg-navbar__cta svg {
            width: 13px;
            height: 13px;
          }
        }

        @media (max-width: 380px) {
          .vg-navbar__inner {
            padding: 0 0.75rem;
          }
          .vg-navbar__brand {
            gap: 0.4rem;
          }
          .vg-navbar__logo-text {
            font-size: 0.9rem;
          }
          .vg-navbar__logo-badge {
            display: none;
          }
          .vg-navbar__cta {
            padding: 7px 14px;
            font-size: 0.65rem;
          }
        }
      `}</style>
    </>
  );
}
