"""
Download Indian language human voice samples from snorbyte/indic-text-audio-sample.
Enhanced with:
- STREAMING with ROBUST RESUME and RETRY capabilities
- Gender diversity enforcement
- Quality filters (duration, SNR)
- Compliance: Age verification through dataset metadata
- Speaker diversity (max samples per speaker)
Downloads Telugu, Malayalam, Hindi, Tamil.
Targets: 200 Train + 50 Test per language (120 train + 30 test more needed)
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
LANGUAGES = {
    "telugu": "te",
    "malayalam": "ml", 
    "hindi": "hi",
    "tamil": "ta"
}

TRAIN_PER_LANG = 200  # Target: 100 male + 100 female
TEST_PER_LANG = 50    # Target: 25 male + 25 female
TARGET_SR = 16000
MIN_DURATION_SEC = 2.0   # Min 2 seconds
MAX_DURATION_SEC = 30    # Max 30 seconds
MIN_SNR_DB = 12.0        # Minimum signal-to-noise ratio
MAX_PER_SPEAKER = 8      # Max samples per speaker for diversity

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
train_human_dir = os.path.join(BASE_DIR, "data", "train", "human")
test_human_dir = os.path.join(BASE_DIR, "data", "test", "human")
os.makedirs(train_human_dir, exist_ok=True)
os.makedirs(test_human_dir, exist_ok=True)


def estimate_snr(audio, sr):
    """Estimate Signal-to-Noise Ratio in dB."""
    try:
        energy = np.sum(audio ** 2)
        if energy == 0:
            return 0
        
        # Use first and last 10% as potential noise
        noise_portion = int(len(audio) * 0.1)
        if noise_portion < 100:
            return 20  # Too short to estimate, assume acceptable
        
        noise = np.concatenate([audio[:noise_portion], audio[-noise_portion:]])
        noise_energy = np.sum(noise ** 2) / len(noise)
        


def normalize_gender(raw_gender):
    """Normalize gender string from dataset."""
    if not raw_gender:
        return None
    g = raw_gender.lower().strip()
    if g in ["male", "man", "m"]:
        return "male"
    elif g in ["female", "woman", "f"]:
        return "female"
    return None
        if noise_energy == 0:
            return 60  # Very clean
        
        snr = 10 * np.log10(energy / (noise_energy * len(audio)))
        return snr
    except:
        return 20  # Default acceptable value

def get_existing_counts():
    """Count files already on disk to enable resuming."""
    counts = defaultdict(lambda: {"male": 0, "female": 0})
    
    # Check both Train and Test directories
    for d in [train_human_dir, test_human_dir]:
        # Pattern: lang_gender_human_*.wav
        for f in70)
    print("Indian Languages Human Voice Download (Telugu, Malayalam, Hindi, Tamil)")
    print("=" * 70)
    
    # Track speakers for diversity
    speaker_counts = defaultdict(int)
    
    retry_count = 0
    max_retries = 5
    
    while retry_count < max_retries:
        try:
            # 1. Check status from disk
            existing_counts = get_existing_counts()
            
            # Check completion
            total_needed = (TRAIN_PER_LANG + TEST_PER_LANG) // 2  # 125 per gender
            all_complete = True
            print("\n--- Current Status ---")
            for lang, code in LANGUAGES.items():
                m_count = existing_counts[code]['male']
                f_count = existing_counts[code]['female']
                total = m_count + f_count
                target_total = TRAIN_PER_LANG + TEST_PER_LANG
                print(f"{lang.capitalize():12s} ({code}): Male {m_count:3d}/{total_needed}, Female {f_count:3d}/{total_needed} | Total: {total}/{target_total}")
                if m_count < total_needed or f_count < total_needed:
                    all_complete = False
            
            if all_complete:
                print("\n✓ All targets reached! Download complete.")
                break

            print(f"\nStarting data stream (attempt {retry_count + 1}/{max_retries})...")
            print("Note: May take time to skip existing files...\n")
            
            # 2. Start Stream with manual decoding for robustness
            ds = load_dataset("snorbyte/indic-text-audio-sample", split="samples", streaming=True)
            ds = ds.cast_column("audio", Audio(decode=False))
            
            # Track "seen valid" items in THIS stream run
            seen_valid_counts = defaultdict(lambda: {"male": 0, "female": 0})
            
            processed_raw = 0
            saved_this_run = 0
            
            for item in ds:
                processed_raw += 1
                if processed_raw % 100 == 0:
                    print(f"Scanning: {processed_raw} items | Saved: {saved_this_run}", end='\r')
                
                # Quick metadata filters
                lang_name = item.get("language", "")
                if lang_name not in LANGUAGES:
                    continue
                
                raw_gender = item.get("user_gender", "")
                gender = normalize_gender(raw_gender)
                if not gender:
                    continue
                
                lang_code = LANGUAGES[lang_name]
                
                # Check if still needed
                current_on_disk = existing_counts[lang_code][gender]
                if current_on_disk >= total_needed:
                    continue  # Already full for this category
                
                # Speaker diversity check
                speaker_id = item.get("user_id", "unknown")
                if speaker_counts[f"{lang_code}_{speaker_id}"] >= MAX_PER_SPEAKER:
                    continue
                
                # Age compliance check (if metadata available)
                age = item.get("user_age")
                if age and age < 18:
                    continue  # Skip minors for compliance
                
                # Decode and process audio
                try:
                    audio_info = item["audio"]
                    audio_bytes = audio_info.get("bytes")
                    if not audio_bytes:
                        continue

                    # Decode audio
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(audio_bytes)
                        tmp_path = tmp.name
                    
                    try:
                        y, sr = librosa.load(tmp_path, sr=TARGET_SR, mono=True)
                    except:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                        continue
                    
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    
                    # Duration filter
                    duration = len(y) / TARGET_SR
                    if duration < MIN_DURATION_SEC:
                        continue
                    
                    # Quality filter: SNR
                    snr = estimate_snr(y, TARGET_SR)
                    if snr < MIN_SNR_DB:
                        continue
                    
                    # VALID sample
                    seen_valid_counts[lang_code][gender] += 1
                    
                    # Resume logic: skip if already on disk
                    if seen_valid_counts[lang_code][gender] <= current_on_disk:
                        continue
                    
                    # NEW file - process and save
                    
                    # Trim if too long
                    if duration > MAX_DURATION_SEC:
                        y = y[:int(MAX_DURATION_SEC * TARGET_SR)]
                        duration = MAX_DURATION_SEC
                    
                    # Normalize audio
                    y = librosa.util.normalize(y)
                    
                    # Determine split (train or test)
                    total_count = seen_valid_counts[lang_code][gender]
                    train_target = TRAIN_PER_LANG // 2
                    
                    if total_count <= train_target:
                        output_dir = train_human_dir
                        file_idx = total_count - 1
                        split_label = "Train"
                    else:
                        output_dir = test_human_dir
                        file_idx = total_count - train_target - 1
                        split_label = "Test"
                    
                    filename = f"{lang_code}_{gender}_human_{file_idx:04d}.wav"
                    out_path = os.path.join(output_dir, filename)
                    sf.write(out_path, y, TARGET_SR)
                    
                    # Update trackers
                    existing_counts[lang_code][gender] += 1
                    speaker_counts[f"{lang_code}_{speaker_id}"] += 1
                    saved_this_run += 1
                    
                    print(f"[{split_label}] {filename} | {duration:.1f}s, SNR:{snr:.1f}dB | {lang_name.capitalize()} M:{existing_counts[lang_code]['male']} F:{existing_counts[lang_code]['female']}          ")

                except Exception as e:
                    print(f"Sample error: {e}")
                    continue

            # End of stream
            print(f"\nStream exhausted. Saved {saved_this_run} files this run.")
            
            # Check if complete
            final_counts = get_existing_counts()
            still_needed = False
            for lang, code in LANGUAGES.items():
                if final_counts[code]['male'] < total_needed or final_counts[code]['female'] < total_needed:
                    still_needed = True
                    break
            
            if not still_needed:
                print("\n✓ All targets reached!")
                break
            else:
                print("\nTargets not yet met. If dataset is exhausted, this is the maximum available.")
                break

        except Exception as e:
            retry_count += 1
            print(f"\n⚠ ERROR: {e}")
            if retry_count < max_retries:
                print(f"Retrying in 10 seconds... ({retry_count}/{max_retries})")
                time.sleep(10)
            else:
                print("Max retries reached. Stopping.")
                break

    # Final summary
    print("\n" + "=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    final_counts = get_existing_counts()
    for lang, code in LANGUAGES.items():
        m = final_counts[code]['male']
        f = final_counts[code]['female']
        total = m + f
        target = TRAIN_PER_LANG + TEST_PER_LANG
        print(f"{lang.capitalize():12s}: Male={m}, Female={f}, Total={total}/{target}")
    print(f"\nUnique speakers used: {len(speaker_counts)}")


if __name__ == "__main__":
    try:
        run_downloader()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_excof loop
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
