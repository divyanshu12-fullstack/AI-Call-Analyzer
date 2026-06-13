'use client';

import { useVoxStore } from '@/lib/store';

const LANGUAGES = [
  { value: 'English', label: 'English' },
  { value: 'Hindi', label: 'Hindi' },
  { value: 'Tamil', label: 'Tamil' },
  { value: 'Telugu', label: 'Telugu' },
  { value: 'Malayalam', label: 'Malayalam' },
] as const;

export default function LanguageSelector() {
  const { language, setLanguage } = useVoxStore();

  return (
    <div style={{ width: '100%' }}>
      <span
        className="text-label"
        style={{ display: 'block', marginBottom: '0.75rem' }}
      >
        Language
      </span>
      <div
        style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '0.5rem',
        }}
      >
        {LANGUAGES.map((lang) => {
          const isSelected = language === lang.value;
          return (
            <button
              key={lang.value}
              type="button"
              onClick={() => setLanguage(lang.value)}
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 'var(--text-mono-size)',
                padding: '0.5rem 1rem',
                borderRadius: '9999px',
                border: `1px solid ${isSelected ? 'var(--accent-signal)' : 'var(--border-subtle)'}`,
                background: isSelected ? 'var(--bg-glass)' : 'transparent',
                color: isSelected ? 'var(--text-primary)' : 'var(--text-secondary)',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                minHeight: '44px',
                backdropFilter: isSelected ? 'blur(8px)' : 'none',
              }}
            >
              {lang.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
