'use client';

import { useCallback, useState } from 'react';
import { useDropzone, type FileRejection } from 'react-dropzone';
import { Upload, FileAudio, AlertCircle } from 'lucide-react';
import { useVoxStore } from '@/lib/store';
import { formatFileSize } from '@/lib/audioUtils';
import WaveformPlayer from './WaveformPlayer';

const ACCEPTED_TYPES: Record<string, string[]> = {
  'audio/mpeg': ['.mp3'],
  'audio/wav': ['.wav'],
  'audio/x-wav': ['.wav'],
  'audio/wave': ['.wav'],
};

export default function FileDropzone() {
  const { file, setFile } = useVoxStore();
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback(
    (acceptedFiles: File[], rejections: FileRejection[]) => {
      setError(null);

      if (rejections.length > 0) {
        setError('Invalid format — only .mp3 and .wav files are accepted');
        return;
      }

      if (acceptedFiles.length > 0) {
        setFile(acceptedFiles[0]);
      }
    },
    [setFile]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxFiles: 1,
    multiple: false,
  });

  const hasFile = file !== null;
  const hasError = error !== null;

  return (
    <div style={{ width: '100%' }}>
      {/* Drop zone - hidden when file is loaded */}
      {!hasFile && (
        <div
          {...getRootProps()}
          style={{
            border: `2px dashed ${
              hasError
                ? 'var(--state-ai)'
                : isDragActive
                  ? 'var(--accent-signal)'
                  : 'var(--border-subtle)'
            }`,
            borderRadius: '12px',
            padding: '3rem 2rem',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '1rem',
            cursor: 'pointer',
            transition: 'border-color 0.2s ease, background 0.2s ease',
            background: isDragActive
              ? 'rgba(56, 189, 248, 0.05)'
              : hasError
                ? 'rgba(248, 113, 113, 0.03)'
                : 'transparent',
            minHeight: '200px',
          }}
        >
          <input {...getInputProps()} />

          {hasError ? (
            <AlertCircle size={32} style={{ color: 'var(--state-ai)' }} />
          ) : (
            <Upload
              size={32}
              style={{
                color: isDragActive
                  ? 'var(--accent-signal)'
                  : 'var(--text-muted)',
              }}
            />
          )}

          <span
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 'var(--text-mono-size)',
              color: hasError
                ? 'var(--state-ai)'
                : isDragActive
                  ? 'var(--accent-signal)'
                  : 'var(--text-muted)',
              textAlign: 'center',
            }}
          >
            {hasError
              ? error
              : isDragActive
                ? 'Release to load audio'
                : 'Drop .mp3 or .wav'}
          </span>
        </div>
      )}

      {/* File loaded state */}
      {hasFile && (
        <div
          style={{
            background: 'var(--bg-glass)',
            border: '1px solid var(--border-subtle)',
            borderRadius: '12px',
            padding: '1.5rem',
          }}
        >
          {/* File info */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.75rem',
              marginBottom: '1rem',
            }}
          >
            <FileAudio size={18} style={{ color: 'var(--accent-signal)' }} />
            <span
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 'var(--text-mono-size)',
                color: 'var(--text-secondary)',
              }}
            >
              {file.name}
            </span>
            <span
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 'var(--text-mono-size)',
                color: 'var(--text-muted)',
                marginLeft: 'auto',
              }}
            >
              {formatFileSize(file.size)}
            </span>
          </div>

          {/* Waveform */}
          <WaveformPlayer file={file} />

          {/* Replace file */}
          <div
            {...getRootProps()}
            style={{
              marginTop: '1rem',
              textAlign: 'center',
              cursor: 'pointer',
            }}
          >
            <input {...getInputProps()} />
            <span
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize: '0.7rem',
                color: 'var(--text-muted)',
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
              }}
            >
              Click to replace file
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
