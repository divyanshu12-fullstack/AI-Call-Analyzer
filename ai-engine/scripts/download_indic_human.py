"""Download Indian-language human voice samples with streaming, filters, and resume support."""
import glob
import os
import tempfile
import time
from collections import defaultdict

import librosa
import numpy as np
import soundfile as sf
from dataset import Audio, load_dataset

LANGUAGE_MAP = {
    "telugu": "te",
    "malayalam": "ml",
    "hindi": "hi",
    "tamil": "ta",
}

TRAIN_PER_LANG = 200
TEST_PER_LANG = 50
TARGET_SR = 16000
MIN_DURATION_SEC = 2.0
MAX_DURATION_SEC = 30.0
MIN_SNR_DB = 12.0
MAX_SPEAKER_SAMPLES = 8
MAX_RETRIES = 5

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train", "human")
TEST_DIR = os.path.join(BASE_DIR, "data", "test", "human")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)


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


def normalize_gender(raw_gender: str) -> str | None:
    if not raw_gender:
        return None
    g = raw_gender.lower()
    if g in {"male", "man", "m"}:
        return "male"
    if g in {"female", "woman", "f"}:
        return "female"
    return None


def count_existing() -> dict:
    counts = defaultdict(lambda: {"male": 0, "female": 0})
    for directory in (TRAIN_DIR, TEST_DIR):
        for path in glob.glob(os.path.join(directory, "*_human_*.wav")):
            parts = os.path.basename(path).split("_")
            if len(parts) < 3:
                continue
            lang_code, gender = parts[0], parts[1]
            if lang_code in LANGUAGE_MAP.values() and gender in {"male", "female"}:
                counts[lang_code][gender] += 1
    return counts


def save_sample(audio: np.ndarray, lang_code: str, gender: str, split: str, index: int) -> None:
    directory = TRAIN_DIR if split == "train" else TEST_DIR
    filename = f"{lang_code}_{gender}_human_{index:04d}.wav"
    path = os.path.join(directory, filename)
    sf.write(path, audio, TARGET_SR)


def ensure_dirs() -> None:
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)


def run_downloader() -> None:
    ensure_dirs()
    speaker_usage = defaultdict(int)
    retry = 0

    while retry < MAX_RETRIES:
        saved = 0
        try:
            existing_counts = count_existing()
            needed_per_gender = (TRAIN_PER_LANG + TEST_PER_LANG) // 2
            targets_met = True
            print(f"\nStatus check #{retry + 1}")
            for lang, code in LANGUAGE_MAP.items():
                m = existing_counts[code]["male"]
                f = existing_counts[code]["female"]
                total = m + f
                print(f"{lang.capitalize():9s} ({code}): Male {m}/{needed_per_gender}, Female {f}/{needed_per_gender} | Total {total}/{TRAIN_PER_LANG + TEST_PER_LANG}")
                if m < needed_per_gender or f < needed_per_gender:
                    targets_met = False
            if targets_met:
                print("Targets satisfied. Download complete.")
                break

            ds = load_dataset("snorbyte/indic-text-audio-sample", split="samples", streaming=True)
            ds = ds.cast_column("audio", Audio(decode=False))
            processed = 0

            for item in ds:
                processed += 1
                if processed % 200 == 0:
                    print(f"Processed {processed} stream items | Saved {saved}", end="\r")

                lang_name = item.get("language")
                if lang_name not in LANGUAGE_MAP:
                    continue
                gender = normalize_gender(item.get("user_gender"))
                if gender is None:
                    continue
                lang_code = LANGUAGE_MAP[lang_name]
                current_count = existing_counts[lang_code][gender]
                if current_count >= needed_per_gender:
                    continue
                speaker_key = f"{lang_code}_{item.get('user_id', 'unknown')}"
                if speaker_usage[speaker_key] >= MAX_SPEAKER_SAMPLES:
                    continue
                user_age = item.get("user_age")
                if user_age and user_age < 18:
                    continue

                audio_data = item.get("audio", {})
                audio_bytes = audio_data.get("bytes")
                if not audio_bytes:
                    continue

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(audio_bytes)
                    tmp_path = tmp.name
                try:
                    y, _ = librosa.load(tmp_path, sr=TARGET_SR, mono=True)
                except Exception:
                    os.remove(tmp_path)
                    continue
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

                duration = len(y) / TARGET_SR
                if not (MIN_DURATION_SEC <= duration <= MAX_DURATION_SEC):
                    continue
                snr = estimate_snr(y)
                if snr < MIN_SNR_DB:
                    continue

                y = librosa.util.normalize(y)
                seen = current_count + 1
                split = "train" if seen <= (TRAIN_PER_LANG // 2) else "test"
                index = seen - 1 if split == "train" else seen - (TRAIN_PER_LANG // 2) - 1
                save_sample(y, lang_code, gender, split, index)
                existing_counts[lang_code][gender] += 1
                speaker_usage[speaker_key] += 1
                saved += 1

            print(f"\nStream run complete. Saved {saved} files.")
            if saved == 0:
                print("No new samples found this round; retrying after a pause...")
                time.sleep(10)
            else:
                time.sleep(2)
        except Exception as exc:
            print(f"\nCRITICAL ERROR: {exc}")
            print("Restarting stream in 5 seconds...")
            time.sleep(5)
        finally:
            retry += 1

    final_counts = count_existing()
    print("\nFinal summary:")
    for lang, code in LANGUAGE_MAP.items():
        m = final_counts[code]["male"]
        f = final_counts[code]["female"]
        print(f"{lang.capitalize():9s} ({code}): Male={m}, Female={f}, Total={m + f}")
    print(f"Unique speakers used: {len([k for k, v in speaker_usage.items() if v > 0])}")


if __name__ == "__main__":
    run_downloader()
