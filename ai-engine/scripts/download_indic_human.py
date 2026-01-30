"""
Download Indian language human voice samples from snorbyte/indic-text-audio-sample.
Uses STREAMING with ROBUST RESUME and RETRY capabilities.
Downloads Telugu, Malayalam, Hindi, Tamil with gender balance.
Max audio duration: 30 seconds.
Enforces 80 Train + 20 Test per language (Total 400 Indian + 100 English = 500).
"""
import os
import glob
import time
import soundfile as sf
import numpy as np
import librosa
import tempfile
from datasets import load_dataset, Audio
from collections import defaultdict

# Configuration
# Note: Lowercase keys as observed in dataset metadata
LANGUAGES = {
    "telugu": "te",
    "malayalam": "ml", 
    "hindi": "hi",
    "tamil": "ta"
}

TRAIN_PER_LANG = 80  # 40 male + 40 female
TEST_PER_LANG = 20   # 10 male + 10 female
TARGET_SR = 16000
MAX_DURATION_SEC = 30  # Max 30 seconds

# Create folders
train_human_dir = "data/train/human"
test_human_dir = "data/test/human"
os.makedirs(train_human_dir, exist_ok=True)
os.makedirs(test_human_dir, exist_ok=True)

def get_existing_counts():
    """Count files already on disk to enable resuming."""
    counts = defaultdict(lambda: {"male": 0, "female": 0})
    
    # Check both Train and Test directories
    for d in [train_human_dir, test_human_dir]:
        # Pattern: lang_gender_human_*.wav
        for f in glob.glob(os.path.join(d, "*_*_human_*.wav")):
            filename = os.path.basename(f)
            parts = filename.split('_')
            # Expect: [lang, gender, 'human', idx.wav]
            if len(parts) >= 3:
                lang_code = parts[0]
                gender = parts[1]
                if lang_code in LANGUAGES.values() and gender in ["male", "female"]:
                    counts[lang_code][gender] += 1
    return counts

def run_downloader():
    print("=" * 60)
    print("Persistent Indian Language Downloader Started")
    print("=" * 60)
    
    while True:
        try:
            # 1. Check status from disk
            existing_counts = get_existing_counts()
            
            # Check completion
            total_needed = (TRAIN_PER_LANG + TEST_PER_LANG) // 2  # 50 per gender
            all_complete = True
            print("\n--- Status ---")
            for lang, code in LANGUAGES.items():
                m_count = existing_counts[code]['male']
                f_count = existing_counts[code]['female']
                print(f"{lang} ({code}): Male {m_count}/{total_needed}, Female {f_count}/{total_needed}")
                if m_count < total_needed or f_count < total_needed:
                    all_complete = False
            
            if all_complete:
                print("\n✅ All targets reached! Exiting.")
                break

            print("\nStarting Stream... (It may take time to skip existing files)")
            
            # 2. Start Stream
            # Timeout options to reduce crashes
            ds = load_dataset("snorbyte/indic-text-audio-sample", split="samples", streaming=True)
            ds = ds.cast_column("audio", Audio(decode=False)) # Manual decode for robustness logic
            
            # Variables to track "seen valid" items in THIS stream run
            # to map to "existing" items on disk 1-to-1.
            seen_valid_counts = defaultdict(lambda: {"male": 0, "female": 0})
            
            processed_raw = 0
            
            for item in ds:
                processed_raw += 1
                if processed_raw % 500 == 0:
                    print(f"Scanning... Raw Items: {processed_raw}", end='\r')
                
                # Metadata Filtering (Fast)
                lang_name = item.get("language", "")
                if lang_name not in LANGUAGES:
                    continue
                
                raw_gender = (item.get("user_gender") or "").lower()
                if raw_gender in ["male", "man"]:
                    gender = "male"
                elif raw_gender in ["female", "woman"]:
                    gender = "female"
                else:
                    continue
                
                lang_code = LANGUAGES[lang_name]
                
                # Check if we still need this category
                current_on_disk = existing_counts[lang_code][gender]
                target_for_cat = (TRAIN_PER_LANG + TEST_PER_LANG) // 2
                
                if current_on_disk >= target_for_cat:
                    # Allow optimization: if we have enough on disk, we don't even need to 
                    # check duration or increment 'seen'. We can completely ignore.
                    # This speeds up "Resume" massively once a category is full.
                    continue

                # Resume Logic:
                # We need to know if this item corresponds to one we ALREADY have on disk.
                # Since we filter by duration, we MUST check duration to count it as "seen valid".
                
                try:
                    audio_info = item["audio"]
                    audio_bytes = audio_info.get("bytes")
                    if not audio_bytes: continue

                    # Decode
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(audio_bytes)
                        tmp_path = tmp.name
                    
                    try:
                        y, sr = librosa.load(tmp_path, sr=TARGET_SR, mono=True)
                    except:
                        if os.path.exists(tmp_path): os.remove(tmp_path)
                        continue
                    
                    if os.path.exists(tmp_path): os.remove(tmp_path)
                    
                    # Duration Filter
                    duration = len(y) / TARGET_SR
                    if duration < 1.0:
                        continue # Skip short
                    
                    # It is VALID.
                    seen_valid_counts[lang_code][gender] += 1
                    
                    # Now check if this valid item is already on disk
                    # 'seen_valid_counts' is 1-based index of valid items in stream.
                    # If seen_valid_counts[n] <= current_on_disk, we have it.
                    
                    if seen_valid_counts[lang_code][gender] <= current_on_disk:
                        # We have this file. Skip writing.
                        if seen_valid_counts[lang_code][gender] % 10 == 0:
                             print(f"Skipping known duplicate {lang_code}/{gender} (#{seen_valid_counts[lang_code][gender]})", end='\r')
                        continue
                    
                    # If we are here, it's a NEW file!
                    # Process and Save.
                    
                    # Trim if needed
                    if duration > MAX_DURATION_SEC:
                        y = y[:int(MAX_DURATION_SEC * TARGET_SR)]
                    
                    # Determine split (Train or Test?)
                    # Total valid seen so far matches (expected) total count on disk + 1.
                    total_count = seen_valid_counts[lang_code][gender] 
                    
                    train_target = TRAIN_PER_LANG // 2
                    
                    if total_count <= train_target:
                        output_dir = train_human_dir
                        # Index is 0-based
                        file_idx = total_count - 1
                    else:
                        output_dir = test_human_dir
                        file_idx = total_count - train_target - 1
                    
                    filename = f"{lang_code}_{gender}_human_{file_idx:04d}.wav"
                    out_path = os.path.join(output_dir, filename)
                    sf.write(out_path, y, TARGET_SR)
                    
                    # Update internal 'existing' tracker so we don't rely only on disk scan till next restart
                    existing_counts[lang_code][gender] += 1
                    
                    print(f"saved: {filename} ({duration:.1f}s) | Total {lang_name} M:{existing_counts[lang_code]['male']} F:{existing_counts[lang_code]['female']}")

                except Exception as e:
                     print(f"Sample Error: {e}")
                     continue

            # End of loop
            print("\nStream finished (Source Exhausted).")
            # If we get here and targets are not met, the dataset didn't have enough.
            break

        except Exception as e:
            print(f"\nCRITICAL ERROR: {e}")
            print("Restarting stream in 5 seconds...")
            time.sleep(5)
            continue

if __name__ == "__main__":
    run_downloader()
