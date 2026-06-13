'use client';

import Link from 'next/link';

export default function Navbar() {
  return (
    <nav
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        height: '56px',
        background: 'rgba(7, 10, 16, 0.8)',
        backdropFilter: 'blur(16px)',
        WebkitBackdropFilter: 'blur(16px)',
        borderBottom: '1px solid var(--border-subtle)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 2rem',
        zIndex: 100,
      }}
    >
      {/* Wordmark */}
      <Link
        href="/"
        style={{
          fontFamily: 'var(--font-display)',
          fontWeight: 700,
          fontSize: '1.1rem',
          color: 'var(--text-primary)',
          textDecoration: 'none',
          letterSpacing: '0.04em',
        }}
      >
        VOXGUARD
      </Link>

      {/* CTA */}
      <Link
        href="/analyze"
        style={{
          background: 'rgba(56, 189, 248, 0.08)',
          border: '1px solid rgba(56, 189, 248, 0.35)',
          color: 'var(--text-primary)',
          fontFamily: 'var(--font-mono)',
          fontSize: '0.75rem',
          fontWeight: 600,
          letterSpacing: '0.06em',
          padding: '8px 20px',
          borderRadius: '100px',
          textDecoration: 'none',
          transition: 'border-color 0.2s, background 0.2s, box-shadow 0.2s',
          minHeight: '36px',
          display: 'inline-flex',
          alignItems: 'center',
        }}
        onMouseEnter={(e) => {
          const el = e.currentTarget;
          el.style.borderColor = 'var(--accent-signal)';
          el.style.background = 'rgba(56, 189, 248, 0.14)';
          el.style.boxShadow = '0 0 24px rgba(56, 189, 248, 0.2)';
        }}
        onMouseLeave={(e) => {
          const el = e.currentTarget;
          el.style.borderColor = 'rgba(56, 189, 248, 0.35)';
          el.style.background = 'rgba(56, 189, 248, 0.08)';
          el.style.boxShadow = 'none';
        }}
      >
        Analyze Audio
      </Link>
    </nav>
  );
}
