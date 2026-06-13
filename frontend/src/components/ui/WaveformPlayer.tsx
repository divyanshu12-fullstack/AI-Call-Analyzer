'use client';

import { useRef, useEffect, useState, useCallback } from 'react';
import WaveSurfer from 'wavesurfer.js';
import { Play, Pause } from 'lucide-react';

interface WaveformPlayerProps {
  file: File;
}

export default function WaveformPlayer({ file }: WaveformPlayerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    if (!containerRef.current) return;

    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: 'rgba(148, 163, 184, 0.4)',
      progressColor: '#38BDF8',
      cursorColor: '#818CF8',
      cursorWidth: 2,
      barWidth: 2,
      barGap: 2,
      barRadius: 2,
      height: 80,
      normalize: true,
      backend: 'WebAudio',
    });

    ws.on('ready', () => setIsReady(true));
    ws.on('play', () => setIsPlaying(true));
    ws.on('pause', () => setIsPlaying(false));
    ws.on('finish', () => setIsPlaying(false));

    const url = URL.createObjectURL(file);
    ws.load(url).catch((err) => {
      if (err.name === 'AbortError') {
        // Safe to ignore when the component is unmounted or file changes
        return;
      }
      console.error('WaveSurfer load error:', err);
    });

    wavesurferRef.current = ws;

    return () => {
      try {
        ws.destroy();
      } catch (e) {
        // Prevent AbortError or signal errors from bubble-crashing the app
        console.warn('WaveSurfer cleanup warning:', e);
      }
      URL.revokeObjectURL(url);
      wavesurferRef.current = null;
      setIsReady(false);
      setIsPlaying(false);
    };
  }, [file]);

  const handlePlayPause = useCallback(() => {
    wavesurferRef.current?.playPause();
  }, []);

  return (
    <div>
      <div
        ref={containerRef}
        style={{
          width: '100%',
          borderRadius: '8px',
          overflow: 'hidden',
          opacity: isReady ? 1 : 0.3,
          transition: 'opacity 0.3s ease',
        }}
      />

      {isReady && (
        <div
          style={{
            display: 'flex',
            justifyContent: 'center',
            marginTop: '0.75rem',
          }}
        >
          <button
            type="button"
            onClick={handlePlayPause}
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: '44px',
              height: '44px',
              borderRadius: '50%',
              border: '1px solid var(--border-subtle)',
              background: 'var(--bg-glass)',
              color: 'var(--accent-signal)',
              cursor: 'pointer',
              transition: 'border-color 0.2s ease',
            }}
            aria-label={isPlaying ? 'Pause audio' : 'Play audio'}
          >
            {isPlaying ? <Pause size={18} /> : <Play size={18} />}
          </button>
        </div>
      )}
    </div>
  );
}
