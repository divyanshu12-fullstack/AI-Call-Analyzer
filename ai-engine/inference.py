"""
Inference module for Voice Detection.
Loads trained model and processes audio for prediction.
Supports confidence calibration via temperature scaling.
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "voice_model_best.pth")
CALIBRATION_PATH = os.path.join(BASE_DIR, "models", "calibration.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class VoiceDetector:
    def __init__(self, model_path=MODEL_PATH, calibration_path=CALIBRATION_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = VoiceResNet().to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        self.model.eval()
        
        # Load calibration if available
        self.temperature = 1.0
        if os.path.exists(calibration_path):
            try:
                calib_data = torch.load(calibration_path, map_location=DEVICE, weights_only=True)
                self.temperature = calib_data.get('temperature', 1.0)
                print(f"Loaded calibration temperature: {self.temperature:.4f}")
            except Exception as e:
                print(f"Warning: Could not load calibration: {e}")
        
    def _compute_all_features(self, audio, sr):
        """Compute all 8 features for the given audio array."""
        # Mel spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
        mel = librosa.power_to_db(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        # MFCCs (13 coefficients + deltas)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc = np.concatenate([mfcc, mfcc_delta], axis=0)  # shape: (26, T)
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)

        # Pitch (F0) using pyin for speed
        f0, _, _ = librosa.pyin(y=audio, sr=sr, fmin=50, fmax=500)
        pitch_track = np.nan_to_num(f0)
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

        # Ensure all features have the exact same number of frames
        min_len = min(
            mel.shape[1], mfcc.shape[1], pitch_track.shape[1],
            contrast.shape[1], chroma.shape[1], zcr.shape[1],
            centroid.shape[1], bandwidth.shape[1]
        )
        
        features = np.concatenate([
            mel[:, :min_len],
            mfcc[:, :min_len],
            pitch_track[:, :min_len],
            contrast[:, :min_len],
            chroma[:, :min_len],
            zcr[:, :min_len],
            centroid[:, :min_len],
            bandwidth[:, :min_len]
        ], axis=0)  # shape: (113, min_len)
        
        return features

    def _extract_features(self, audio_path):
        """Extract all audio features and concatenate them (padded to MAX_LEN)."""
        audio, sr = librosa.load(audio_path, sr=16000)
        features = self._compute_all_features(audio, sr)
        
        # Pad or trim to MAX_LEN
        if features.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - features.shape[1]
            features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
        else:
            features = features[:, :MAX_LEN]
            
        return features
    
    def _apply_calibration(self, prob):
        """Apply temperature scaling calibration to raw probability."""
        if self.temperature != 1.0:
            # Convert probability to logit-like space, apply temperature, convert back
            logit = np.log(prob / (1 - prob + 1e-10) + 1e-10)
            calibrated_logit = logit / self.temperature
            prob = 1.0 / (1.0 + np.exp(-calibrated_logit))
        return np.clip(prob, 1e-15, 1 - 1e-15)
    
    def _generate_explanation(self, confidence, is_ai, calibrated=False):
        """
        Generate a detailed technical explanation for the prediction.
        Includes confidence level and specific artifacts detected.
        """
        confidence_level = "very high" if confidence > 0.9 else "high" if confidence > 0.75 else "moderate" if confidence > 0.6 else "low"
        calibration_note = " (confidence calibrated)" if calibrated and self.temperature != 1.0 else ""
        
        if is_ai:
            explanations = {
                0.9: f"Very high confidence ({confidence:.2%}) - Clear synthetic speech patterns detected. High spectral uniformity, consistent pitch, lack of natural breath sounds and micro-pauses strongly indicate AI-generated audio.{calibration_note}",
                0.75: f"High confidence ({confidence:.2%}) - Strong evidence of synthetic generation. Detected unnaturally smooth spectral transitions, rigid formant patterns, and absence of organic speech artifacts.{calibration_note}",
                0.6: f"Moderate confidence ({confidence:.2%}) - Likely AI-generated. Detected synthetic artifacts in frequency domain including unnatural pitch consistency and spectral smoothing.{calibration_note}",
                0.0: f"Low confidence ({confidence:.2%}) - Some indicators of synthetic speech detected, but confidence is borderline. Audio may be lightly processed AI or natural with artifacts.{calibration_note}"
            }
        else:
            explanations = {
                0.9: f"Very high confidence ({confidence:.2%}) - Clear human speech patterns. Natural pitch variations, presence of breath sounds, organic spectral characteristics, and natural temporal dynamics indicate human origin.{calibration_note}",
                0.75: f"High confidence ({confidence:.2%}) - Strong indicators of natural human speech. Detected natural formant transitions, irregular pitch patterns, and organic speech rhythm characteristic of human vocalization.{calibration_note}",
                0.6: f"Moderate confidence ({confidence:.2%}) - Likely human speech. Detected natural prosody patterns, breath artifacts, and organic spectral irregularities consistent with human voice.{calibration_note}",
                0.0: f"Low confidence ({confidence:.2%}) - Some human characteristics present but confidence is borderline. Audio may be heavily processed human or realistic AI-generated speech.{calibration_note}"
            }
        
        # Select explanation based on confidence threshold
        if confidence > 0.9:
            return explanations[0.9]
        elif confidence > 0.75:
            return explanations[0.75]
        elif confidence > 0.6:
            return explanations[0.6]
        else:
            return explanations[0.0]
    
    def _extract_windows(self, audio_path, window_sec=5.0, stride_sec=2.5):
        """Extract multiple 5-second feature windows from long audio efficiently."""
        audio, sr = librosa.load(audio_path, sr=16000)
        
        features = self._compute_all_features(audio, sr)
        total_frames = features.shape[1]
        
        # Approximate frame conversions (librosa default hop_length=512)
        hop_length = 512
        stride_frames = int((stride_sec * sr) / hop_length)
        
        windows = []
        
        if total_frames < MAX_LEN:
            # Pad and take one
            pad_width = MAX_LEN - total_frames
            feat = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
            windows.append(feat)
        else:
            # Sliding window over the features array
            for start in range(0, total_frames - MAX_LEN + 1, stride_frames):
                segment = features[:, start : start + MAX_LEN]
                windows.append(segment)
            
            # Ensure we don't miss tail end
            if (total_frames - MAX_LEN) % stride_frames > (MAX_LEN // 2):
                segment = features[:, -MAX_LEN:]
                windows.append(segment)

        return np.array(windows)

    def predict_from_file(self, audio_path):
        """Predict using a sliding window to handle long dialogues."""
        try:
            # Extract all windows (e.g., 30s audio -> ~10-12 overlapping windows)
            windows = self._extract_windows(audio_path)
            windows_tensor = torch.tensor(windows, dtype=torch.float32).to(DEVICE)
            
            with torch.no_grad():
                # Model handles batch of windows efficiently
                probs = self.model(windows_tensor).squeeze(-1).cpu().numpy()
            
            # If probs is a single value (one window), wrap it in a list
            if probs.ndim == 0:
                probs = np.array([probs])
                
            # Calibrate all probabilities
            calibrated_probs = [self._apply_calibration(p) for p in probs]
            
            # Aggregation Strategy:
            # 1. If any window is EXTREMELY likely AI (e.g. > 90%), flag as AI.
            # 2. Otherwise, take the mean.
            max_prob = max(calibrated_probs)
            avg_prob = sum(calibrated_probs) / len(calibrated_probs)
            
            if max_prob > 0.90:
                final_prob = max_prob
            else:
                final_prob = avg_prob
                
            is_ai = final_prob > 0.5
            confidence = final_prob if is_ai else (1 - final_prob)
            
            return {
                "classification": "AI_GENERATED" if is_ai else "HUMAN",
                "confidence": round(float(confidence), 4),
                "explanation": self._generate_explanation(confidence, is_ai, calibrated=(self.temperature != 1.0)),
                "meta": {
                    "windows_analyzed": len(calibrated_probs),
                    "max_ai_prob": round(float(max_prob), 4),
                    "avg_ai_prob": round(float(avg_prob), 4)
                }
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            raise e    
    def _is_mp3_bytes(self, audio_bytes: bytes) -> bool:
        if not audio_bytes or len(audio_bytes) < 4:
            return False
        # ID3 tag header or MPEG frame sync
        if audio_bytes.startswith(b"ID3"):
            return True
        return audio_bytes[0] == 0xFF and (audio_bytes[1] & 0xE0) == 0xE0

    def predict_from_base64(self, audio_base64: str):
        """Predict from base64-encoded audio (MP3 only)."""
        if not audio_base64:
            raise ValueError("audioBase64 is required")

        try:
            audio_bytes = base64.b64decode(audio_base64, validate=True)
        except Exception:
            raise ValueError("Invalid Base64 audio data")

        if not self._is_mp3_bytes(audio_bytes):
            raise ValueError("Only MP3 audio is supported")

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
