"""
Download human voice samples from ai4bharat/IndicVoices dataset for Indian languages.
Languages: Hindi, Malayalam, Telugu, Tamil
Train: 200 samples per language
Test: 50 samples per language
Max duration: 20 seconds

This script downloads parquet files directly from HuggingFace Hub and processes them.
"""

import os
import glob
import io
import tempfile
from collections import defaultdict
import numpy as np
import soundfile as sf
import librosa
import pyarrow.parquet as pq
import pandas as pd

# HuggingFace imports
from huggingface_hub import hf_hub_download, list_repo_tree

# Configuration
LANGUAGES = {
    "hindi": "hindi",
    "malayalam": "malayalam", 
    "telugu": "telugu",
    "tamil": "tamil"
}

TRAIN_TARGET = 200
TEST_TARGET = 50
TARGET_SR = 16000
MIN_DURATION_SEC = 1.0
MAX_DURATION_SEC = 20.0  # Maximum 20 seconds as per requirement
MIN_SNR_DB = 10.0
MAX_PER_SPEAKER = 10  # Limit samples per speaker for diversity

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train", "human")
TEST_DIR = os.path.join(BASE_DIR, "data", "test", "human")

# Create directories
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)


def count_existing(prefix: str, directory: str) -> int:
    """Count existing files with given language prefix."""
    pattern = os.path.join(directory, f"{prefix}_*_human_*.wav")
    return len(glob.glob(pattern))


def estimate_snr(audio: np.ndarray) -> float:
    """Estimate Signal-to-Noise Ratio."""
    energy = np.sum(audio ** 2)
    if energy <= 0:
        return 0.0
    noise_portion = max(1, int(len(audio) * 0.1))
    noise = np.concatenate([audio[:noise_portion], audio[-noise_portion:]])
    noise_energy = np.sum(noise ** 2) / len(noise)
    if noise_energy <= 0:
        return 60.0
    return 10 * np.log10(energy / (noise_energy * len(audio)))


def decode_audio_from_bytes(audio_bytes):
    """Decode audio from bytes using soundfile or librosa."""
    if audio_bytes is None or len(audio_bytes) == 0:
        return None, None
    
    # Ensure bytes type
    if hasattr(audio_bytes, 'tobytes'):
        audio_bytes = audio_bytes.tobytes()
    elif not isinstance(audio_bytes, bytes):
        try:
            audio_bytes = bytes(audio_bytes)
        except:
            return None, None
    
    # Try different audio formats
    for suffix in ['.wav', '.flac', '.mp3', '.ogg', '.opus']:
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            
            audio, sample_rate = librosa.load(tmp_path, sr=None, mono=True)
            os.unlink(tmp_path)
            return audio.astype(np.float32), sample_rate
        except Exception as e:
            try:
                os.unlink(tmp_path)
            except:
                pass
            continue
    
    # Try soundfile directly
    try:
        with io.BytesIO(audio_bytes) as audio_io:
            audio, sample_rate = sf.read(audio_io)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            return audio.astype(np.float32), sample_rate
    except:
        pass
    
    return None, None


def process_audio(audio, original_sr: int):
    """Process audio: resample, normalize, and validate."""
    try:
        # Handle stereo to mono
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Ensure float32
        if audio.dtype != np.float32:
            if audio.dtype in [np.int16, np.int32]:
                audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
            else:
                audio = audio.astype(np.float32)
        
        # Resample if needed
        if original_sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=TARGET_SR)
        
        # Calculate duration
        duration = len(audio) / TARGET_SR
        
        # Duration check
        if not (MIN_DURATION_SEC <= duration <= MAX_DURATION_SEC):
            return None, duration, "duration"
        
        # SNR check
        snr = estimate_snr(audio)
        if snr < MIN_SNR_DB:
            return None, 0, "snr"
        
        # Normalize
        audio = librosa.util.normalize(audio)
        
        return audio, duration, "ok"
        
    except Exception as e:
        return None, 0, f"error: {e}"


def get_parquet_files(language_name: str):
    """Get list of parquet files for a language."""
    try:
        files = list(list_repo_tree(
            'ai4bharat/IndicVoices', 
            path_in_repo=language_name, 
            repo_type='dataset'
        ))
        # Filter for train parquet files (we'll use train split which has more data)
        train_files = [f.path for f in files if 'train-' in f.path and f.path.endswith('.parquet')]
        valid_files = [f.path for f in files if 'valid-' in f.path and f.path.endswith('.parquet')]
        return sorted(train_files), sorted(valid_files)
    except Exception as e:
        print(f"Error listing files: {e}")
        return [], []


def download_and_process_language(language_code: str, language_name: str):
    """Download and process samples for a specific language."""
    print(f"\n{'='*60}")
    print(f"Processing {language_name.upper()}")
    print(f"{'='*60}")
    
    # Count existing samples
    existing_train = count_existing(language_code, TRAIN_DIR)
    existing_test = count_existing(language_code, TEST_DIR)
    train_needed = max(0, TRAIN_TARGET - existing_train)
    test_needed = max(0, TEST_TARGET - existing_test)
    
    print(f"Current {language_name} counts:")
    print(f"  Train: {existing_train}/{TRAIN_TARGET}")
    print(f"  Test : {existing_test}/{TEST_TARGET}")
    
    if train_needed == 0 and test_needed == 0:
        print(f"Targets already satisfied for {language_name}. Skipping.")
        return existing_train, existing_test
    
    train_saved = 0
    test_saved = 0
    speaker_counts = defaultdict(int)
    
    # Get list of parquet files
    train_files, valid_files = get_parquet_files(language_name)
    print(f"  Found {len(train_files)} train files, {len(valid_files)} valid files")
    
    # Process train files first (they have more samples)
    all_files = train_files + valid_files
    
    if not all_files:
        print(f"  No parquet files found for {language_name}")
        return existing_train, existing_test
    
    for file_path in all_files:
        if train_saved >= train_needed and test_saved >= test_needed:
            break
            
        try:
            print(f"\n  Downloading {file_path}...")
            
            # Download the parquet file
            parquet_path = hf_hub_download(
                repo_id="ai4bharat/IndicVoices",
                filename=file_path,
                repo_type="dataset"
            )
            
            # Read the parquet file
            table = pq.read_table(parquet_path)
            df = table.to_pandas()
            
            print(f"  Processing {len(df)} samples from {file_path.split('/')[-1]}")
            
            skipped = {"duration": 0, "snr": 0, "error": 0, "speaker": 0, "decode": 0}
            file_train = 0
            file_test = 0
            
            for idx, row in df.iterrows():
                if train_saved >= train_needed and test_saved >= test_needed:
                    break
                
                try:
                    # Check duration first (metadata check is faster)
                    duration = row.get('duration', 0)
                    if pd.notna(duration) and (duration < MIN_DURATION_SEC or duration > MAX_DURATION_SEC):
                        skipped["duration"] += 1
                        continue
                    
                    # Extract audio bytes from audio_filepath column
                    audio_filepath = row.get('audio_filepath', None)
                    
                    if audio_filepath is None:
                        skipped["error"] += 1
                        continue
                    
                    # Handle the audio_filepath structure
                    audio_bytes = None
                    if isinstance(audio_filepath, dict):
                        audio_bytes = audio_filepath.get('bytes', None)
                    elif hasattr(audio_filepath, '__getitem__'):
                        try:
                            audio_bytes = audio_filepath['bytes']
                        except:
                            pass
                    
                    if audio_bytes is None:
                        skipped["decode"] += 1
                        continue
                    
                    # Decode audio
                    audio, sample_rate = decode_audio_from_bytes(audio_bytes)
                    
                    if audio is None:
                        skipped["decode"] += 1
                        continue
                    
                    # Process the audio
                    processed_audio, duration, status = process_audio(audio, sample_rate)
                    
                    if processed_audio is None:
                        if "duration" in status:
                            skipped["duration"] += 1
                        elif "snr" in status:
                            skipped["snr"] += 1
                        else:
                            skipped["error"] += 1
                        continue
                    
                    # Get speaker ID for diversity
                    speaker_id = row.get('speaker_id', str(idx))
                    if pd.isna(speaker_id):
                        speaker_id = str(idx)
                    if speaker_counts[speaker_id] >= MAX_PER_SPEAKER:
                        skipped["speaker"] += 1
                        continue
                    
                    # Determine gender
                    gender = row.get('gender', 'unknown')
                    if pd.isna(gender) or gender not in ["male", "female", "Male", "Female"]:
                        gender = "mixed"
                    else:
                        gender = gender.lower()
                    
                    # Save to train or test
                    if train_saved < train_needed:
                        file_idx = existing_train + train_saved
                        out_path = os.path.join(
                            TRAIN_DIR, 
                            f"{language_code}_{gender}_human_{file_idx:04d}.wav"
                        )
                        sf.write(out_path, processed_audio, TARGET_SR)
                        speaker_counts[speaker_id] += 1
                        train_saved += 1
                        file_train += 1
                        if train_saved % 50 == 0 or train_saved == train_needed:
                            print(f"    [Train] Progress: {train_saved}/{train_needed}")
                            
                    elif test_saved < test_needed:
                        file_idx = existing_test + test_saved
                        out_path = os.path.join(
                            TEST_DIR, 
                            f"{language_code}_{gender}_human_{file_idx:04d}.wav"
                        )
                        sf.write(out_path, processed_audio, TARGET_SR)
                        speaker_counts[speaker_id] += 1
                        test_saved += 1
                        file_test += 1
                        if test_saved % 25 == 0 or test_saved == test_needed:
                            print(f"    [Test] Progress: {test_saved}/{test_needed}")
                
                except Exception as e:
                    skipped["error"] += 1
                    continue
            
            print(f"  File stats - Train: +{file_train}, Test: +{file_test}, Skipped: {skipped}")
            
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            continue
    
    final_train = existing_train + train_saved
    final_test = existing_test + test_saved
    
    print(f"\n{language_name} summary:")
    print(f"  Train: {final_train}/{TRAIN_TARGET}")
    print(f"  Test : {final_test}/{TEST_TARGET}")
    print(f"  Unique speakers: {len(speaker_counts)}")
    
    return final_train, final_test


def main():
    """Main entry point."""
    print("=" * 60)
    print("IndicVoices Human Voice Downloader")
    print("Languages: Hindi, Malayalam, Telugu, Tamil")
    print(f"Train target: {TRAIN_TARGET} per language")
    print(f"Test target: {TEST_TARGET} per language")
    print(f"Max duration: {MAX_DURATION_SEC}s")
    print("=" * 60)
    
    print("\nNote: This script downloads from HuggingFace Hub.")
    print("Make sure you have run: huggingface-cli login")
    
    # Suppress HuggingFace symlink warning
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    results = {}
    
    for lang_code, lang_name in LANGUAGES.items():
        try:
            train_count, test_count = download_and_process_language(lang_code, lang_name)
            results[lang_name] = {"train": train_count, "test": test_count}
        except Exception as e:
            print(f"Error processing {lang_name}: {e}")
            import traceback
            traceback.print_exc()
            results[lang_name] = {"train": 0, "test": 0, "error": str(e)}
    
    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for lang_name, counts in results.items():
        print(f"{lang_name.capitalize()}:")
        print(f"  Train: {counts.get('train', 0)}/{TRAIN_TARGET}")
        print(f"  Test : {counts.get('test', 0)}/{TEST_TARGET}")
        if "error" in counts:
            print(f"  Error: {counts['error']}")
    
    total_train = sum(c.get('train', 0) for c in results.values())
    total_test = sum(c.get('test', 0) for c in results.values())
    print(f"\nTotal files:")
    print(f"  Train: {total_train}/{TRAIN_TARGET * len(LANGUAGES)}")
    print(f"  Test : {total_test}/{TEST_TARGET * len(LANGUAGES)}")


if __name__ == "__main__":
    main()
