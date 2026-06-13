import type { Metadata } from 'next';
import { Inter, JetBrains_Mono } from 'next/font/google';
import './globals.css';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
  weight: ['300', '400', '500', '600'],
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jetbrains',
  display: 'swap',
  weight: ['400', '500', '600', '700'],
});

export const metadata: Metadata = {
  title: 'VoxGuard AI — Deepfake Speech Detection',
  description:
    'Detect AI-generated speech with SE-VoiceResNet-18. 11-channel feature extraction, sliding window inference, and calibrated confidence scoring across 5 languages.',
  keywords: [
    'deepfake detection',
    'AI voice detection',
    'synthetic speech',
    'VoxGuard',
    'audio forensics',
    'SE-VoiceResNet',
  ],
  openGraph: {
    title: 'VoxGuard AI — Deepfake Speech Detection',
    description:
      'Professional-grade synthetic speech detection powered by SE-VoiceResNet-18 with 97% accuracy across 5 languages.',
    type: 'website',
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <body>{children}</body>
    </html>
  );
}
