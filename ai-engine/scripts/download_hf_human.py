"""Download English human voice samples (LibriSpeech dev-clean) with resume and quality checks."""
import glob
import os
import tarfile
import urllib.request
from collections import defaultdict

import librosa
import numpy as np
import soundfile as sf

TRAIN_TARGET = 200
TEST_TARGET = 50
TARGET_SR = 16000
MIN_DURATION_SEC = 2.0
MAX_DURATION_SEC = 30.0
MIN_SNR_DB = 15.0
MAX_PER_SPEAKER = 6

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train", "human")
TEST_DIR = os.path.join(BASE_DIR, "data", "test", "human")
TEMP_DIR = os.path.join(BASE_DIR, "temp_download")
TAR_FILE = os.path.join(TEMP_DIR, "dev-clean.tar.gz")
EXTRACT_DIR = os.path.join(TEMP_DIR, "LibriSpeech")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


def count_existing(prefix: str, directory: str) -> int:
    pattern = os.path.join(directory, f"{prefix}_*_human_*.wav")
    return len(glob.glob(pattern))


def estimate_snr(audio: np.ndarray) -> float:
    energy = np.sum(audio ** 2)
    if energy <= 0:
        return 0.0
    noise_portion = max(1, int(len(audio) * 0.1))
    noise = np.concatenate([audio[:noise_portion], audio[-noise_portion:]])
    noise_energy = np.sum(noise ** 2) / len(noise)
    if noise_energy <= 0:
        return 60.0
    return 10 * np.log10(energy / (noise_energy * len(audio)))


def download_archive() -> None:
    if os.path.exists(TAR_FILE):
        print("Using cached LibriSpeech archive.")
        return
    print("Downloading LibriSpeech dev-clean (~337 MB)...")
    urllib.request.urlretrieve(LIBRISPEECH_URL, TAR_FILE)


def extract_archive() -> None:
    if os.path.exists(EXTRACT_DIR) and os.listdir(EXTRACT_DIR):
        print("Archive already extracted.")
        return
    print("Extracting archive...")
    with tarfile.open(TAR_FILE, "r:gz") as tar:
        tar.extractall(TEMP_DIR)


def process_file(flac_path: str, train_needed: int, test_needed: int, speaker_counts: dict, current_train: int, current_test: int):
    try:
        audio, _ = librosa.load(flac_path, sr=TARGET_SR, mono=True)
    except Exception:
        return 0, 0

    duration = len(audio) / TARGET_SR
    if not (MIN_DURATION_SEC <= duration <= MAX_DURATION_SEC):
        return 0, 0

    snr = estimate_snr(audio)
    if snr < MIN_SNR_DB:
        return 0, 0

    audio = librosa.util.normalize(audio)
    parts = flac_path.split(os.sep)
    speaker_id = next((p for p in parts if p.isdigit()), None)
    if speaker_id and speaker_counts[speaker_id] >= MAX_PER_SPEAKER:
        return 0, 0

    gender = "female" if speaker_id and int(speaker_id) % 2 == 0 else "male"
    if current_train < train_needed:
        out_path = os.path.join(TRAIN_DIR, f"english_{gender}_human_{current_train:04d}.wav")
        sf.write(out_path, audio, TARGET_SR)
        if speaker_id:
            speaker_counts[speaker_id] += 1
        print(f"[Train] Saved {current_train + 1}/{train_needed} ({duration:.1f}s, SNR={snr:.1f}dB)")
        return 1, 0
    if current_test < test_needed:
        out_path = os.path.join(TEST_DIR, f"english_{gender}_human_{current_test:04d}.wav")
        sf.write(out_path, audio, TARGET_SR)
        if speaker_id:
            speaker_counts[speaker_id] += 1
        print(f"[Test] Saved {current_test + 1}/{test_needed} ({duration:.1f}s, SNR={snr:.1f}dB)")
        return 0, 1
    return 0, 0


def main() -> None:
    existing_train = count_existing("english", TRAIN_DIR)
    existing_test = count_existing("english", TEST_DIR)
    train_needed = max(0, TRAIN_TARGET - existing_train)
    test_needed = max(0, TEST_TARGET - existing_test)

    print("Current English human counts:")
    print(f"  Train: {existing_train}/{TRAIN_TARGET}")
    print(f"  Test : {existing_test}/{TEST_TARGET}")

    if train_needed == 0 and test_needed == 0:
        print("Targets already satisfied. Nothing to download.")
        return

    download_archive()
    extract_archive()

    flac_files = sorted(glob.glob(os.path.join(EXTRACT_DIR, "**", "*.flac"), recursive=True))
    train_saved = 0
    test_saved = 0
    speaker_counts = defaultdict(int)

    for flac_path in flac_files:
        if train_saved >= train_needed and test_saved >= test_needed:
            break
        t_inc, s_inc = process_file(
            flac_path,
            train_needed,
            test_needed,
            speaker_counts,
            existing_train + train_saved,
            existing_test + test_saved,
        )
        train_saved += t_inc
        test_saved += s_inc

    print("\nDownload finished.")
    print(f"  Train total: {existing_train + train_saved}/{TRAIN_TARGET}")
    print(f"  Test total : {existing_test + test_saved}/{TEST_TARGET}")
    print(f"  Unique speakers used this run: {len([s for s, count in speaker_counts.items() if count > 0])}")


if __name__ == "__main__":
    main()
