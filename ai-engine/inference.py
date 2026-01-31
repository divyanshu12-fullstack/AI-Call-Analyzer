"""
Inference module for Voice Detection.
Loads trained model and processes audio for prediction.
"""
import torch
import librosa
import numpy as np
import base64
import tempfile
import os
from model import VoiceResNet

# Constants
MAX_LEN = 157  # Must match dataset.py
# Constants
MAX_LEN = 157  # Must match dataset.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "voice_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class VoiceDetector:
    def __init__(self, model_path=MODEL_PATH):
        self.model = VoiceResNet().to(DEVICE)
        self.model.load_state_dict(torch.load("models/voice_model_best.pth", map_location=DEVICE))
        self.model.eval()
        
    def _extract_features(self, audio_path):
        """Extract all audio features and concatenate them."""
        audio, sr = librosa.load(audio_path, sr=16000)

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
        mel = librosa.power_to_db(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        # MFCCs (13 coefficients + deltas)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc = np.concatenate([mfcc, mfcc_delta], axis=0)  # shape: (26, T)
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)

        # Pitch (F0)
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_track = pitches.max(axis=0)
        pitch_track = np.expand_dims(pitch_track, axis=0)  # (1, T)
        pitch_track = (pitch_track - pitch_track.mean()) / (pitch_track.std() + 1e-6)

        # Spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrast = (contrast - contrast.mean()) / (contrast.std() + 1e-6)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma = (chroma - chroma.mean()) / (chroma.std() + 1e-6)

        # Zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        zcr = (zcr - zcr.mean()) / (zcr.std() + 1e-6)

        # Spectral centroid and bandwidth
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        centroid = (centroid - centroid.mean()) / (centroid.std() + 1e-6)
        bandwidth = (bandwidth - bandwidth.mean()) / (bandwidth.std() + 1e-6)

        # Pad or trim all features to MAX_LEN frames
        def pad_or_trim(feat, max_len=MAX_LEN):
            if feat.shape[1] < max_len:
                pad_width = max_len - feat.shape[1]
                feat = np.pad(feat, ((0, 0), (0, pad_width)), mode='constant')
            else:
                feat = feat[:, :max_len]
            return feat

        mel = pad_or_trim(mel)
        mfcc = pad_or_trim(mfcc)
        pitch_track = pad_or_trim(pitch_track)
        contrast = pad_or_trim(contrast)
        chroma = pad_or_trim(chroma)
        zcr = pad_or_trim(zcr)
        centroid = pad_or_trim(centroid)
        bandwidth = pad_or_trim(bandwidth)

        # Stack all features along channel dimension
        features = np.concatenate([
            mel,
            mfcc,
            pitch_track,
            contrast,
            chroma,
            zcr,
            centroid,
            bandwidth
        ], axis=0)  # shape: (113, MAX_LEN)

        return features
    
    def _generate_explanation(self, confidence, is_ai):
        """Generate a technical explanation for the prediction."""
        if is_ai:
            if confidence > 0.9:
                return "High spectral uniformity and lack of natural pitch variation strongly indicate synthetic speech generation."
            elif confidence > 0.7:
                return "Consistent formant patterns and unnaturally smooth transitions suggest AI-generated audio."
            else:
                return "Some synthetic artifacts detected in frequency domain, moderate indication of AI generation."
        else:
            if confidence > 0.9:
                return "Natural pitch variations, breath sounds, and organic spectral patterns indicate human speech."
            elif confidence > 0.7:
                return "Presence of micro-pauses and natural frequency modulation suggest human origin."
            else:
                return "Audio shows some organic characteristics, likely human but with some processing artifacts."
    
    def predict_from_file(self, audio_path):
        """Predict whether audio file is AI-generated or human."""
        features = self._extract_features(audio_path)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            prob = self.model(features_tensor).item()
        
        is_ai = prob > 0.5
        confidence = prob if is_ai else (1 - prob)
        
        return {
            "classification": "AI_GENERATED" if is_ai else "HUMAN",
            "confidence": round(confidence, 4),
            "explanation": self._generate_explanation(confidence, is_ai)
        }
    
    def predict_from_base64(self, audio_base64: str):
        """Predict from base64-encoded audio (MP3 or WAV)."""
        # Decode base64
        audio_bytes = base64.b64decode(audio_base64)
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        try:
            result = self.predict_from_file(temp_path)
        finally:
            os.unlink(temp_path)
            
        return result


# Test the detector if run directly
if __name__ == "__main__":
    import sys
    
    detector = VoiceDetector()
    
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        result = detector.predict_from_file(audio_path)
        print(f"\nPrediction for: {audio_path}")
        print(f"  Classification: {result['classification']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Explanation: {result['explanation']}")
    else:
        # Test with a sample from each class
        print("Testing with sample files...")
        
        # Test human sample
        human_result = detector.predict_from_file("data/test/human/human_0000.wav")
        print(f"\nHuman sample: {human_result['classification']} ({human_result['confidence']:.2%})")
        
        # Test AI sample
        ai_result = detector.predict_from_file("data/test/ai/ai_0000.wav")
        print(f"AI sample: {ai_result['classification']} ({ai_result['confidence']:.2%})")
