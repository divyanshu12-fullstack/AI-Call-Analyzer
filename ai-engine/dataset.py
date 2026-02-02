
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
            
            # Walk through directory tree to find all .wav files (handles nested subdirectories)
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith(".wav"):
                        self.files.append(os.path.join(root, file))
                        self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def _augment_audio(self, audio):
        """Apply random augmentation to audio during training."""
        # Time stretching (0.8x - 1.2x speed)
        if np.random.rand() < 0.3:
            rate = np.random.uniform(0.8, 1.2)
            audio = librosa.effects.time_stretch(audio, rate=rate)
        
        # Pitch shifting (±2 semitones)
        if np.random.rand() < 0.3:
            n_steps = np.random.randint(-2, 3)
            audio = librosa.effects.pitch_shift(audio, sr=16000, n_steps=n_steps)
        
        # Background noise injection (SNR 10-30 dB)
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 0.005, len(audio))
            snr_db = np.random.uniform(10, 30)
            snr_linear = 10 ** (snr_db / 20)
            audio = audio + noise / snr_linear
            audio = np.clip(audio, -1.0, 1.0)
        
        # Random gain adjustment
        if np.random.rand() < 0.3:
            gain = np.random.uniform(0.8, 1.2)
            audio = audio * gain
            audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def _apply_specaugment(self, mel):
        """Apply SpecAugment: time and frequency masking."""
        # Time masking: mask consecutive time steps
        if np.random.rand() < 0.3:
            t_mask = np.random.randint(10, 30)  # mask up to 30 frames
            t_start = np.random.randint(0, max(1, mel.shape[1] - t_mask))
            mel = mel.copy()
            mel[:, t_start:t_start + t_mask] = 0
        
        # Frequency masking: mask consecutive mel bands
        if np.random.rand() < 0.3:
            f_mask = np.random.randint(5, 15)  # mask up to 15 mel bands
            f_start = np.random.randint(0, max(1, mel.shape[0] - f_mask))
            mel = mel.copy()
            mel[f_start:f_start + f_mask, :] = 0
        
        return mel

    def __getitem__(self, index):
        file_path = self.files[index]
        label = self.labels[index]

        audio, sr = librosa.load(file_path, sr=16000)
        
        # Apply augmentation during training
        if self.augment:
            audio = self._augment_audio(audio)

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
        mel = librosa.power_to_db(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        
        # Apply SpecAugment (time and frequency masking) during training
        if self.augment:
            mel = self._apply_specaugment(mel)

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
        ], axis=0)  # shape: (C, MAX_LEN)

        features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1, C, MAX_LEN)
        features = features.squeeze(0)  # (C, MAX_LEN)

        return features, torch.tensor(label, dtype=torch.float32)
