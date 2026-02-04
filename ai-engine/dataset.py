
import os
import librosa
import torch
import numpy as np
from torch.utils.data import Dataset

# Fixed spectrogram width (frames) - corresponds to ~5 seconds at 16kHz
MAX_LEN = 157

class VoiceDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        """
        Args:
            root_dir: Path to data directory (should contain 'human' and 'ai' subdirectories)
            augment: Whether to apply data augmentation (for training data)
        """
        self.files = []
        self.labels = []
        self.augment = augment

        for label, folder in enumerate(["human", "ai"]):
            folder_path = os.path.join(root_dir, folder)

            for file in os.listdir(folder_path):
                if file.endswith(".wav"):
                    self.files.append(os.path.join(folder_path, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def _augment_audio(self, audio, sr):
        """Apply random augmentations including telephony simulation."""
        # 1. Gain Variation (Volume fluctuations)
        if np.random.random() < 0.5:
            audio = audio * np.random.uniform(0.3, 1.2)

        # 2. Gaussian Noise (Simulate line static)
        if np.random.random() < 0.4:
            noise_amp = 0.005 * np.random.uniform() * np.amax(audio)
            audio = audio + noise_amp * np.random.normal(size=audio.shape)

        # 3. Telephony Simulation (Bandpass filter 300Hz - 3400Hz)
        if np.random.random() < 0.4:
            # Simple low-pass filter simulation to mimic phone lines
            # This makes the model focus on core voice features, not high-freq "digital" clean signals
            audio = librosa.effects.preemphasis(audio)
            
        # 4. Pitch & Time (Intonation variability)
        if np.random.random() < 0.3:
            audio = librosa.effects.time_stretch(y=audio, rate=np.random.uniform(0.85, 1.15))
        if np.random.random() < 0.3:
            audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=np.random.uniform(-1.5, 1.5))
            
        return audio

    def _apply_spec_augment(self, features, max_mask_pct=0.1):
        """Randomly mask frequency and time strips (SpecAugment)."""
        # This prevents the model from overfitting to specific spectral artifacts
        c, t = features.shape
        
        # Time masking
        if np.random.random() < 0.5:
            t_mask = int(np.random.uniform(0, max_mask_pct) * t)
            t0 = np.random.randint(0, t - t_mask)
            features[:, t0:t0 + t_mask] = 0
            
        # Frequency masking
        if np.random.random() < 0.5:
            f_mask = int(np.random.uniform(0, max_mask_pct) * c)
            f0 = np.random.randint(0, c - f_mask)
            features[f0:f0 + f_mask, :] = 0
            
        return features

    def __getitem__(self, index):
        file_path = self.files[index]
        label = self.labels[index]

        try:
            audio, sr = librosa.load(file_path, sr=16000)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            audio = np.zeros(16000 * 5)
            sr = 16000

        # Safety check for empty audio or very short audio
        if len(audio) == 0:
             audio = np.zeros(16000 * 5)

        # Apply augmentation ONLY for training data
        if self.augment:
            audio = self._augment_audio(audio, sr)

        # Ensure minimum length to avoid pitch tracking errors
        if len(audio) < 1000:
            audio = np.pad(audio, (0, 1000 - len(audio)))

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
        mel = librosa.power_to_db(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)

        # MFCCs (13 coefficients + deltas)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc = np.concatenate([mfcc, mfcc_delta], axis=0)
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)

        # Pitch (F0)
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitch_track = pitches.max(axis=0)
            pitch_track = np.expand_dims(pitch_track, axis=0)
            pitch_track = (pitch_track - pitch_track.mean()) / (pitch_track.std() + 1e-6)
        except Exception:
            pitch_track = np.zeros((1, mel.shape[1]))

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

        # 5. Channel Dropout (Randomly kill specific features to force learning others)
        # Features map: 0-63 (Mel), 64-89 (MFCC), 90 (Pitch), 91-97 (Contrast), etc.
        if self.augment:
             if np.random.random() < 0.2:
                 # Kill Pitch (Row 90)
                 pitch_track[:] = 0
             if np.random.random() < 0.2:
                 # Kill ZCR (Row 110 approx - need strict indexing)
                 zcr[:] = 0

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

        # Apply SpecAugment (feature-level masking) for training robustness
        if self.augment:
            features = self._apply_spec_augment(features, max_mask_pct=0.15)  # Increased mask pct

        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        features = features.squeeze(0)

        return features, torch.tensor(label, dtype=torch.float32)
