"""
Download multilingual human voice samples from HuggingFace.
Source: snorbyte/indic-text-audio-sample dataset

This downloads human-spoken audio in 5 languages:
- Tamil (ta)
- English (en) - from existing LibriSpeech
- Hindi (hi)
- Malayalam (ml)
- Telugu (te)

The dataset structure includes:
- language: Language code
- audio: Audio bytes
- text: Transcript
- user_age: Speaker age
- user_gender: Speaker gender
"""
import os
import sys
import shutil
import librosa
import soundfile as sf
from huggingface_hub import hf_hub_download
import pandas as pd

# Configuration
SAMPLES_PER_LANGUAGE = 30  # Per language (train + test)
TRAIN_TEST_SPLIT = 0.8  # 80% train, 20% test
TARGET_SR = 16000
MAX_DURATION = 5  # seconds

# Languages we care about (from the 5 required: Tamil, English, Hindi, Malayalam, Telugu)
# Note: English will come from existing LibriSpeech data, Indic languages from HuggingFace
# The HuggingFace dataset uses full language names, not ISO codes
# Based on dataset inspection:
#   - hindi: 416 samples ✓
#   - malayalam: 57 samples ✓
#   - tamil: 1 sample only (insufficient)
#   - telugu: 0 samples (not available)
TARGET_LANGUAGES = {
    "hindi": "hindi",
    "tamil": "tamil",  
    "telugu": "telugu",
    "malayalam": "malayalam",
}

# Directories - organized by language
BASE_DATA_DIR = "data"
TEMP_DIR = "temp_indic_download"

def download_indic_dataset():
    """Download the Indic dataset parquet file from HuggingFace."""
    print("=" * 60)
    print("Downloading Indic Text-Audio Sample Dataset")
    print("Source: snorbyte/indic-text-audio-sample")
    print("=" * 60)
    
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    try:
        repo_id = "snorbyte/indic-text-audio-sample"
        filename = "data_shard_000_zstd.parquet"
        
        print(f"\nDownloading {filename}...")
        print("This may require HuggingFace login if the dataset is gated.")
        
        local_file = hf_hub_download(
            repo_id=repo_id, 
            filename=filename, 
            repo_type="dataset"
        )
        
        print(f"✓ Downloaded to: {local_file}")
        return local_file
        
    except Exception as e:
        print(f"✗ Error downloading dataset: {e}")
        print("\nIf this is a gated dataset, you may need to:")
        print("1. Visit https://huggingface.co/datasets/snorbyte/indic-text-audio-sample")
        print("2. Agree to the dataset terms")
        print("3. Login using: huggingface-cli login")
        return None

def process_audio_bytes(audio_bytes, output_path):
    """
    Process audio bytes: save to temp file, convert to target format, save.
    Returns True if successful, False otherwise.
    """
    import tempfile
    
    try:
        # Save raw bytes to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        # Load and convert with librosa
        audio, sr = librosa.load(temp_path, sr=TARGET_SR, mono=True)
        
        # Trim to max duration
        max_samples = TARGET_SR * MAX_DURATION
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # Skip very short clips (less than 1 second)
        if len(audio) < TARGET_SR:
            os.remove(temp_path)
            return False
        
        # Save processed audio
        sf.write(output_path, audio, TARGET_SR)
        
        # Cleanup temp file
        os.remove(temp_path)
        return True
        
    except Exception as e:
        print(f"  Error processing audio: {e}")
        return False

def extract_samples_by_language(parquet_path):
    """
    Extract audio samples from parquet file, organized by language.
    Creates language-specific folders in data/train/{language}/ and data/test/{language}/
    """
    print("\nLoading parquet file...")
    df = pd.read_parquet(parquet_path)
    
    print(f"Total samples in dataset: {len(df)}")
    
    # Show available languages
    if 'language' in df.columns:
        available_langs = df['language'].value_counts()
        print("\nAvailable languages in dataset:")
        for lang, count in available_langs.items():
            print(f"  {lang}: {count} samples")
    else:
        print("Warning: No 'language' column found. Columns available:", df.columns.tolist())
        return
    
    # Create directories for each target language
    for lang_code, lang_name in TARGET_LANGUAGES.items():
        train_dir = os.path.join(BASE_DATA_DIR, "train", "human", lang_name)
        test_dir = os.path.join(BASE_DATA_DIR, "test", "human", lang_name)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
    
    # Process each target language
    for lang_code, lang_name in TARGET_LANGUAGES.items():
        print(f"\n{'=' * 50}")
        print(f"Processing: {lang_name.upper()} ({lang_code})")
        print(f"{'=' * 50}")
        
        # Filter samples for this language
        lang_df = df[df['language'] == lang_code]
        
        if len(lang_df) == 0:
            print(f"  ⚠ No samples found for language code '{lang_code}'")
            # Try alternative language codes
            alt_codes = [lang_code.lower(), lang_code.upper(), lang_name, lang_name.capitalize()]
            for alt in alt_codes:
                alt_df = df[df['language'] == alt]
                if len(alt_df) > 0:
                    lang_df = alt_df
                    print(f"  ✓ Found {len(lang_df)} samples using code '{alt}'")
                    break
        else:
            print(f"  Found {len(lang_df)} samples")
        
        if len(lang_df) == 0:
            print(f"  ⚠ Skipping {lang_name} - no samples available")
            continue
        
        # Limit to needed samples
        samples_needed = min(SAMPLES_PER_LANGUAGE, len(lang_df))
        lang_df = lang_df.head(samples_needed)
        
        # Calculate train/test split
        train_count = int(samples_needed * TRAIN_TEST_SPLIT)
        test_count = samples_needed - train_count
        
        train_dir = os.path.join(BASE_DATA_DIR, "train", "human", lang_name)
        test_dir = os.path.join(BASE_DATA_DIR, "test", "human", lang_name)
        
        processed = 0
        train_saved = 0
        test_saved = 0
        
        for idx, row in lang_df.iterrows():
            try:
                # Extract audio bytes
                audio_data = row.get('audio', {})
                if isinstance(audio_data, dict):
                    audio_bytes = audio_data.get('bytes', None)
                else:
                    audio_bytes = audio_data
                
                if audio_bytes is None:
                    continue
                
                # Determine output path (train or test)
                if processed < train_count:
                    output_path = os.path.join(train_dir, f"human_{lang_name}_{processed:04d}.wav")
                    is_train = True
                else:
                    test_idx = processed - train_count
                    output_path = os.path.join(test_dir, f"human_{lang_name}_{test_idx:04d}.wav")
                    is_train = False
                
                # Process and save
                if process_audio_bytes(audio_bytes, output_path):
                    if is_train:
                        train_saved += 1
                        print(f"  [Train] {lang_name}: {train_saved}/{train_count}")
                    else:
                        test_saved += 1
                        print(f"  [Test] {lang_name}: {test_saved}/{test_count}")
                    processed += 1
                
            except Exception as e:
                print(f"  Error processing sample: {e}")
                continue
        
        print(f"\n  ✓ {lang_name.upper()} completed:")
        print(f"    Train samples: {train_saved}")
        print(f"    Test samples: {test_saved}")

def ensure_english_samples():
    """
    Check if English samples exist from LibriSpeech.
    If not, create symlinks or copy from the existing human folder.
    """
    english_train_dir = os.path.join(BASE_DATA_DIR, "train", "human", "english")
    english_test_dir = os.path.join(BASE_DATA_DIR, "test", "human", "english")
    
    os.makedirs(english_train_dir, exist_ok=True)
    os.makedirs(english_test_dir, exist_ok=True)
    
    # Check if we already have English samples from existing LibriSpeech download
    existing_train = os.path.join(BASE_DATA_DIR, "train", "human")
    existing_test = os.path.join(BASE_DATA_DIR, "test", "human")
    
    # Copy existing human samples as English
    print("\n" + "=" * 50)
    print("Organizing English samples")
    print("=" * 50)
    
    train_count = 0
    test_count = 0
    
    # Copy train samples
    if os.path.exists(existing_train):
        for f in os.listdir(existing_train):
            if f.endswith('.wav') and f.startswith('human_'):
                src = os.path.join(existing_train, f)
                dst = os.path.join(english_train_dir, f.replace('human_', 'human_english_'))
                if os.path.isfile(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    train_count += 1
    
    # Copy test samples  
    if os.path.exists(existing_test):
        for f in os.listdir(existing_test):
            if f.endswith('.wav') and f.startswith('human_'):
                src = os.path.join(existing_test, f)
                dst = os.path.join(english_test_dir, f.replace('human_', 'human_english_'))
                if os.path.isfile(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    test_count += 1
    
    print(f"  English train samples: {train_count}")
    print(f"  English test samples: {test_count}")
    
    if train_count == 0 and test_count == 0:
        print("  ⚠ No existing English samples found.")
        print("  Run 'python download_hf_human.py' first to get LibriSpeech samples.")

def main():
    """Main execution flow."""
    print("\n" + "=" * 70)
    print("  MULTILINGUAL HUMAN VOICE SAMPLE DOWNLOADER")
    print("  Languages: Tamil, English, Hindi, Malayalam, Telugu")
    print("=" * 70)
    
    # Step 1: Download Indic dataset
    parquet_path = download_indic_dataset()
    
    if parquet_path:
        # Step 2: Extract samples by language
        extract_samples_by_language(parquet_path)
    
    # Step 3: Ensure English samples are organized
    ensure_english_samples()
    
    # Step 4: Print summary
    print("\n" + "=" * 70)
    print("  DOWNLOAD SUMMARY")
    print("=" * 70)
    
    for lang_name in ["tamil", "english", "hindi", "malayalam", "telugu"]:
        train_dir = os.path.join(BASE_DATA_DIR, "train", "human", lang_name)
        test_dir = os.path.join(BASE_DATA_DIR, "test", "human", lang_name)
        
        train_count = len([f for f in os.listdir(train_dir) if f.endswith('.wav')]) if os.path.exists(train_dir) else 0
        test_count = len([f for f in os.listdir(test_dir) if f.endswith('.wav')]) if os.path.exists(test_dir) else 0
        
        status = "✓" if train_count > 0 else "✗"
        print(f"  {status} {lang_name.capitalize():12} - Train: {train_count:3}, Test: {test_count:3}")
    
    print("\n" + "=" * 70)
    print("  Next step: Run 'python generate_multilingual_ai.py' to generate AI voices")
    print("=" * 70)
    
    # Cleanup
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

if __name__ == "__main__":
    main()
