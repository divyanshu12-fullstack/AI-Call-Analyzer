"""
Download human voice samples from OpenSLR (Open Speech and Language Resources).
This uses direct HTTP downloads to avoid HuggingFace audio decoding issues.
"""
import os
import urllib.request
import tarfile
import shutil
import librosa
import soundfile as sf
import glob

# Configuration
TRAIN_COUNT = 100
TEST_COUNT = 20
TARGET_SR = 16000

# Create folders
train_human_dir = "data/train/human"
test_human_dir = "data/test/human"
temp_dir = "temp_download"

os.makedirs(train_human_dir, exist_ok=True)
os.makedirs(test_human_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

# Download a small sample from LibriSpeech dev-clean (directly from OpenSLR)
# This is a smaller subset that's easy to download
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
TAR_FILE = os.path.join(temp_dir, "dev-clean.tar.gz")

print("Downloading LibriSpeech dev-clean dataset...")
print("This may take a few minutes (337 MB)...")

try:
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

    total_needed = TRAIN_COUNT + TEST_COUNT
    count = 0

    for flac_path in flac_files:
        if count >= total_needed:
            break
            
        try:
            # Load and convert to 16kHz mono WAV
            audio, sr = librosa.load(flac_path, sr=TARGET_SR, mono=True)
            
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
    
finally:
    # Cleanup - remove extracted files but keep tar for future use
    extracted_dir = os.path.join(temp_dir, "LibriSpeech")
    if os.path.exists(extracted_dir):
        print("Cleaning up extracted files...")
        shutil.rmtree(extracted_dir)
