import os
import librosa
import torch
import numpy as np
from torch.utils.data import Dataset

# Fixed spectrogram width (frames) - corresponds to ~5 seconds at 16kHz
MAX_LEN = 157

class VoiceDataset(Dataset):
    def __init__(self, root_dir):
        self.files = []
        self.labels = []

        for label, folder in enumerate(["human", "ai"]):
            folder_path = os.path.join(root_dir, folder)

            for file in os.listdir(folder_path):
                if file.endswith(".wav"):
                    self.files.append(os.path.join(folder_path, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = self.files[index]
        label = self.labels[index]

        audio, sr = librosa.load(file_path, sr=16000)

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=64
        )

        mel = librosa.power_to_db(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)  # Avoid division by zero

        # Pad or trim to fixed length
        if mel.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel = mel[:, :MAX_LEN]

        mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)

        return mel, torch.tensor(label, dtype=torch.float32)
