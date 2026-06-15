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
      timeout: 120000, // 2 minute timeout for long audio files
    }
  );

  return response.data;
}
