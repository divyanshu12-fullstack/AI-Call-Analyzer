import { create } from 'zustand';

export interface AnalysisResult {
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

interface VoxStore {
  file: File | null;
  language: string;
  result: AnalysisResult | null;
  isAnalyzing: boolean;
  error: string | null;

  setFile: (f: File) => void;
  setLanguage: (l: string) => void;
  setResult: (r: AnalysisResult) => void;
  setIsAnalyzing: (v: boolean) => void;
  setError: (e: string | null) => void;
  clearResult: () => void;
  reset: () => void;
}

export const useVoxStore = create<VoxStore>((set) => ({
  file: null,
  language: 'English',
  result: null,
  isAnalyzing: false,
  error: null,

  setFile: (f) => set({ file: f, error: null }),
  setLanguage: (l) => set({ language: l }),
  setResult: (r) => set({ result: r, isAnalyzing: false }),
  setIsAnalyzing: (v) => set({ isAnalyzing: v, error: null }),
  setError: (e) => set({ error: e, isAnalyzing: false }),
  clearResult: () => set({ result: null }),
  reset: () => set({ file: null, language: 'English', result: null, isAnalyzing: false, error: null }),
}));
