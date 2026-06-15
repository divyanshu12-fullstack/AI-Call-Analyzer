import axios from 'axios';

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'https://ai-call-analyzer.onrender.com';
const API_KEY = process.env.NEXT_PUBLIC_API_KEY ?? '';

export interface AnalysisResponse {
  status: string;
  language: string;
  classification: 'AI_GENERATED' | 'HUMAN' | 'AI';
  confidenceScore: number;
  explanation: string;
  meta: {
    windows_analyzed: number;
    max_ai_prob: number;
    avg_ai_prob: number;
  };
}

export interface AnalysisError {
  status: 'error';
  message: string;
}

/**
 * Pre-warm the backend by pinging the health endpoint.
 * Call this early (e.g. on app load) so Render wakes up
 * before the user uploads audio.
 */
export async function prewarmBackend(): Promise<void> {
  try {
    await axios.head(BASE_URL, { timeout: 10000 });
  } catch {
    // Swallow errors — pre-warm is best-effort
  }
}

export async function analyzeAudio(
  audioBase64: string,
  language: string,
  audioFormat: string = 'mp3'
): Promise<AnalysisResponse> {
  const response = await axios.post<AnalysisResponse>(
    `${BASE_URL}/api/voice-detection`,
    {
      language,
      audioFormat,
      audioBase64,
    },
    {
      headers: {
        'Content-Type': 'application/json',
        ...(API_KEY ? { 'x-api-key': API_KEY } : {}),
      },
      timeout: 180000, // 3 minute timeout — safety margin for cold starts
    }
  );

  return response.data;
}
