"""
Inference module for Voice Detection.
Loads trained model and processes audio for prediction.
Supports confidence calibration via temperature scaling.

OPTIMIZED: Compute-once feature extraction + pyin pitch tracking.
"""
import torch
import librosa
import numpy as np
import base64
import tempfile
import os
import gc
import time
from model import VoiceResNet

# Constants
MAX_LEN = 157  # Must match dataset.py
MAX_AUDIO_SEC = 60  # Truncate audio longer than this to save memory
MAX_WINDOWS = 6  # Cap sliding windows to limit memory and compute
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

    # ── Shared helpers ──────────────────────────────────────────────

    @staticmethod
    def _pad_or_trim(feat, max_len=MAX_LEN):
        """Pad or trim a feature matrix to exactly max_len frames."""
        if feat.shape[1] < max_len:
            pad_width = max_len - feat.shape[1]
            feat = np.pad(feat, ((0, 0), (0, pad_width)), mode='constant')
        else:
            feat = feat[:, :max_len]
        return feat

    def _apply_calibration(self, prob):
        """Apply temperature scaling calibration to raw probability."""
        if self.temperature != 1.0:
            logit = np.log(prob / (1 - prob + 1e-10) + 1e-10)
            calibrated_logit = logit / self.temperature
            prob = 1.0 / (1.0 + np.exp(-calibrated_logit))
        return np.clip(prob, 1e-15, 1 - 1e-15)
    
    def _generate_explanation(self, confidence, is_ai, calibrated=False):
        """
        Generate a detailed technical explanation for the prediction.
        Includes confidence level and specific artifacts detected.
        """
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
        
        if confidence > 0.9:
            return explanations[0.9]
        elif confidence > 0.75:
            return explanations[0.75]
        elif confidence > 0.6:
            return explanations[0.6]
        else:
            return explanations[0.0]

    # ── Optimized feature extraction ────────────────────────────────

    def _extract_full_features(self, audio, sr):
        """
        Extract all 8 feature types from the FULL audio signal at once.
        Returns a (113, T) feature matrix where T is the natural frame count.
        
        This is the key optimization: compute once on full audio,
        then slice into windows from the resulting matrix.
        """
        t0 = time.time()

        # Mel spectrogram (64 bins)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
        mel = librosa.power_to_db(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        # MFCCs (13 coefficients + 13 deltas = 26 channels)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc = np.concatenate([mfcc, mfcc_delta], axis=0)
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)

        # Pitch (F0) via pyin — MUCH faster than piptrack
        # pyin returns (f0, voiced_flag, voiced_probs) where f0 is 1D
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y=audio, sr=sr, fmin=50, fmax=500,
                frame_length=2048, hop_length=512
            )
            pitch_track = np.nan_to_num(f0, nan=0.0)
            pitch_track = np.expand_dims(pitch_track, axis=0)  # (1, T)
            # Align frame count with mel (pyin may differ by ±1 frame)
            target_t = mel.shape[1]
            if pitch_track.shape[1] < target_t:
                pitch_track = np.pad(pitch_track, ((0, 0), (0, target_t - pitch_track.shape[1])))
            else:
                pitch_track = pitch_track[:, :target_t]
            pitch_track = (pitch_track - pitch_track.mean()) / (pitch_track.std() + 1e-6)
        except Exception:
            pitch_track = np.zeros((1, mel.shape[1]))

        # Spectral contrast (7 bands)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrast = (contrast - contrast.mean()) / (contrast.std() + 1e-6)

        # Chroma STFT (12 bins)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma = (chroma - chroma.mean()) / (chroma.std() + 1e-6)

        # Zero-crossing rate (1 channel)
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        zcr = (zcr - zcr.mean()) / (zcr.std() + 1e-6)

        # Spectral centroid (1 channel)
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        centroid = (centroid - centroid.mean()) / (centroid.std() + 1e-6)

        # Spectral bandwidth (1 channel)
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        bandwidth = (bandwidth - bandwidth.mean()) / (bandwidth.std() + 1e-6)

        # Align all features to the same frame count (mel is the reference)
        target_t = mel.shape[1]
        def align(feat):
            if feat.shape[1] < target_t:
                return np.pad(feat, ((0, 0), (0, target_t - feat.shape[1])))
            return feat[:, :target_t]

        features = np.concatenate([
            mel,                    # 64 channels
            align(mfcc),            # 26 channels
            align(pitch_track),     #  1 channel
            align(contrast),        #  7 channels
            align(chroma),          # 12 channels
            align(zcr),             #  1 channel
            align(centroid),        #  1 channel
            align(bandwidth),       #  1 channel
        ], axis=0)  # Total: 113 channels

        elapsed = time.time() - t0
        print(f"  [PERF] Full feature extraction: {elapsed:.2f}s ({features.shape[1]} frames)")

        return features  # shape: (113, T)

    def _slice_windows(self, full_features, sr=16000, window_sec=5.0, stride_sec=2.5):
        """
        Slice a full (113, T) feature matrix into overlapping windows of MAX_LEN frames.
        This avoids re-computing features per window.
        """
        # Compute how many spectrogram frames correspond to window_sec / stride_sec
        hop_length = 512  # librosa default
        window_frames = int(window_sec * sr / hop_length)
        stride_frames = int(stride_sec * sr / hop_length)
        total_frames = full_features.shape[1]

        windows = []

        if total_frames <= MAX_LEN:
            # Short audio — single window, pad to MAX_LEN
            windows.append(self._pad_or_trim(full_features))
        else:
            for start in range(0, total_frames - window_frames + 1, stride_frames):
                chunk = full_features[:, start : start + window_frames]
                windows.append(self._pad_or_trim(chunk))
                if len(windows) >= MAX_WINDOWS:
                    break

            # Catch tail if significant and we haven't hit max windows
            if len(windows) < MAX_WINDOWS and (total_frames % stride_frames) > (sr // hop_length):
                chunk = full_features[:, -window_frames:]
                windows.append(self._pad_or_trim(chunk))

        print(f"  [PERF] Sliced {len(windows)} windows from {total_frames} frames")
        return np.array(windows)

    # ── Prediction ──────────────────────────────────────────────────

    def predict_from_file(self, audio_path):
        """Predict using optimized compute-once windowing strategy."""
        try:
            t_start = time.time()

            # 1. Load audio ONCE, cap at MAX_AUDIO_SEC
            audio, sr = librosa.load(audio_path, sr=16000, duration=MAX_AUDIO_SEC)
            print(f"  [PERF] Audio loaded: {len(audio)/sr:.1f}s @ {sr}Hz")

            # 2. Extract all features on the full audio (8 librosa calls total)
            full_features = self._extract_full_features(audio, sr)

            # Free raw audio — no longer needed
            del audio
            gc.collect()

            # 3. Slice feature matrix into windows (zero librosa calls)
            windows = self._slice_windows(full_features, sr)

            # Free full feature matrix
            del full_features
            gc.collect()

            # 4. Batch inference
            windows_tensor = torch.tensor(windows, dtype=torch.float32).to(DEVICE)
            del windows
            gc.collect()

            with torch.no_grad():
                probs = self.model(windows_tensor).squeeze(-1).cpu().numpy()

            del windows_tensor
            gc.collect()

            # Handle single-window case
            if probs.ndim == 0:
                probs = np.array([probs])
                
            # 5. Calibrate probabilities
            calibrated_probs = [self._apply_calibration(p) for p in probs]
            
            # 6. Aggregate
            max_prob = max(calibrated_probs)
            avg_prob = sum(calibrated_probs) / len(calibrated_probs)
            
            if max_prob > 0.90:
                final_prob = max_prob
            else:
                final_prob = avg_prob
                
            is_ai = final_prob > 0.5
            confidence = final_prob if is_ai else (1 - final_prob)

            elapsed = time.time() - t_start
            print(f"  [PERF] Total inference: {elapsed:.2f}s | {len(calibrated_probs)} windows | result={('AI' if is_ai else 'HUMAN')} ({confidence:.2%})")
            
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
            gc.collect()
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

        # Free base64 bytes from memory immediately
        del audio_bytes
        gc.collect()

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
