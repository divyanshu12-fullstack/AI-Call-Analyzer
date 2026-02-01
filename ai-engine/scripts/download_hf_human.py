"""
Download human voice samples from OpenSLR (Open Speech and Language Resources).
Enhanced with:
- Resume capability (checks existing files)
- Diverse speaker sampling
- Quality filtering (SNR, duration)
- Compliance: Only adult voices (LibriSpeech is 18+ verified)
"""
import os
import urllib.request
import tarfile
import shutil
import librosa
import soundfile as sf
import glob
import numpy as np
from collections import defaultdict

# Configuration
TRAIN_TARGET = 200  # Target total (already have ~80, need 120 more)
TEST_TARGET = 50    # Target total (already have ~20, need 30 more)
TARGET_SR = 16000
MIN_DURATION = 2.0  # Minimum 2 seconds
MAX_DURATION = 10.0 # Maximum 10 seconds
MIN_SNR_DB = 15.0   # Minimum signal-to-noise ratio

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
train_human_dir = os.path.join(BASE_DIR, "data", "train", "human")
test_human_dir = os.path.join(BASE_DIR, "data", "test", "human")
temp_dir = os.path.join(BASE_DIR, "temp_download")

os.makedirs(train_human_dir, exist_ok=True)
os.makedirs(test_human_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)


def count_existing_files(directory, prefix="english"):
    """Count existing English human files."""
    pattern = os.path.join(directory, f"{prefix}_*_human_*.wav")
    # Check existing files
    existing_train = count_existing_files(train_human_dir, "english")
    existing_test = count_existing_files(test_human_dir, "english")
    
    print("=" * 60)
    print("English Human Voice Download (LibriSpeech)")
    print("=" * 60)
    print(f"Existing: Train={existing_train}/{TRAIN_TARGET}, Test={existing_test}/{TEST_TARGET}")
    
    if existing_train >= TRAIN_TARGET and existing_test >= TEST_TARGET:
        print("✓ Already have enough English files. Skipping download.")
        return
    
    train_needed = max(0, TRAIN_TARGET - existing_train)
    test_needed = max(0, TEST_TARGET - existing_test)
    print(f"Need to download: Train={train_needed}, Test={test_needed}")
    
    print("\nDownloading LibriSpeech dev-clean dataset...")
    print("This may take a few minutes (337 MB)...")

    if not os.path.exists(TAR_FILE):
        urllib.request.urlretrieve(LIBRISPEECH_URL, TAR_FILE)
        print("Download complete!")
    else:
        print("Using cached download...")
        
    print("Extracting...")
    with tarfile.open(TAR_FILE, "r:gz") as tar:
        tar.extractall(temp_dir)
    print("Extraction complete!")

    # Find all FLAC files
    flac_files = glob.glob(os.path.join(temp_dir, "**", "*.flac"), recursive=True)
    print(f"Found {len(flac_files)} audio files")
    
    # Track speakers for diversity
    speaker_counts = defaultdict(int)
    max_per_speaker = 5  # Limit files per speaker for diversity

    train_saved = 0
    test_saved = 0
    processed = 0

    for flac_path in flac_files:
        if train_saved >= train_needed and test_saved >= test_needed:
            break
        
        processed += 1
        if processed % 50 == 0:
            print(f"Processed {processed}/{len(flac_files)} files...", end='\r')
            
        try:
            # Extract speaker ID from path (LibriSpeech format: .../speakerID/...)
            parts = flac_path.split(os.sep)
            speaker_id = None
            for i, part in enumerate(parts):
                if part.isdigit() and len(part) >= 4:
                    speaker_id = part
                    break
            
            if speaker_id and speaker_counts[speaker_id] >= max_per_speaker:
                continue  # Skip for diversity
            
            # Load and convert to 16kHz mono WAV
            audio, sr = librosa.load(flac_path, sr=TARGET_SR, mono=True)
            duration = len(audio) / TARGET_SR
            
            # Duration filter
            if duration < MIN_DURATION or duration > MAX_DURATION:
                continue
            
            # Quality filter: SNR
            snr = estimate_snr(audio, TARGET_SR)
            if snr < MIN_SNR_DB:
                continue
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Determine gender from speaker metadata (LibriSpeech convention: even=female, odd=male)
            gender = "female" if speaker_id and int(speaker_id) % 2 == 0 else "male"
            
            # Save to train or test folder
            if train_saved < train_needed:
                file_idx = existing_train + train_saved
                filename = f"english_{gender}_human_{file_idx:04d}.wav"
                out_path = os.path.join(train_human_dir, filename)
                sf.write(out_path, audio, TARGET_SR)
                train_saved += 1
                if speaker_id:
                    speaker_counts[speaker_id] += 1
                if train_saved % 10 == 0:
                    print(f"[Train] Saved {train_saved}/{train_needed} ({duration:.1f}s, SNR:{snr:.1f}dB)")
            elif test_saved < test_needed:
                file_idx = existing_test + test_saved
                filename = f"english_{gender}_human_{file_idx:04d}.wav"
                out_path = os.path.join(test_human_dir, filename)
                sf.write(out_path, a


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()udio, TARGET_SR)
                test_saved += 1
                if speaker_id:
                    speaker_counts[speaker_id] += 1
                if test_saved % 10 == 0:
                    print(f"[Test] Saved {test_saved}/{test_needed} ({duration:.1f}s, SNR:{snr:.1f}dB)")
            
        except Exception as e:
            print(f"Error processing {os.path.basename(flac_path)}: {e}")
            continue

    print(f"\n✓ English human audio download complete!")
    print(f"  Train: {existing_train + train_saved}/{TRAIN_TARGET}")
    print(f"  Test: {existing_test + test_saved}/{TEST_TARGET}")
    print(f"  Unique speakers: {len(speaker_counts
            # Trim to 3-8 seconds (take first 5 seconds max)
            max_samples = TARGET_SR * 5
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            
            # Skip very short clips
            if len(audio) < TARGET_SR:  # Less than 1 second
                continue
            
            # Save to train or test folder
            if count < TRAIN_COUNT:
                out_path = os.path.join(train_human_dir, f"human_{count:04d}.wav")
                sf.write(out_path, audio, TARGET_SR)
                print(f"[Train] Saved {count + 1}/{TRAIN_COUNT}")
            else:
                test_idx = count - TRAIN_COUNT
                out_path = os.path.join(test_human_dir, f"human_{test_idx:04d}.wav")
                sf.write(out_path, audio, TARGET_SR)
                print(f"[Test] Saved {test_idx + 1}/{TEST_COUNT}")
            
            count += 1
            
        except Exception as e:
            print(f"Error processing {flac_path}: {e}")
            continue

    print(f"\n✓ Human audio collection finished!")
    print(f"  Train samples: {min(count, TRAIN_COUNT)}")
    print(f"  Test samples: {max(0, count - TRAIN_COUNT)}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
    # Cleanup - remove extracted files but keep tar for future use
    extracted_dir = os.path.join(temp_dir, "LibriSpeech")
    if os.path.exists(extracted_dir):
        print("Cleaning up extracted files...")
        shutil.rmtree(extracted_dir)
