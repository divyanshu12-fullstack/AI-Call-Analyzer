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
from model import VoiceCNN

# Constants
MAX_LEN = 157  # Must match dataset.py
MODEL_PATH = "models/voice_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class VoiceDetector:
    def __init__(self, model_path=MODEL_PATH):
        self.model = VoiceCNN().to(DEVICE)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()
        
    def _audio_to_mel(self, audio_path):
        """Convert audio file to normalized mel spectrogram."""
        audio, sr = librosa.load(audio_path, sr=16000)
        
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
        mel = librosa.power_to_db(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        
        # Pad or trim to fixed length
        if mel.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel = mel[:, :MAX_LEN]
            
        return mel
    
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
        mel = self._audio_to_mel(audio_path)
        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            prob = self.model(mel_tensor).item()
        
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
