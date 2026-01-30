"""
Cleanup script to enforce strict dataset limits for AI voices.
Ensures exactly 50 Male / 50 Female for training, and ~12/13 for testing per language.
"""
import os
import glob

# Configuration
LANGUAGES = ["te", "ml", "hi", "ta", "en"]
TRAIN_TARGET_PER_GENDER = 50
TEST_TARGET_PER_GENDER = 13  # Approx 12-13

train_dir = "data/train/ai"
test_dir = "data/test/ai"

print("=" * 60)
print("Cleaning AI Voice Dataset")
print("=" * 60)

def cleanup_folder(folder, target_per_gender):
    if not os.path.exists(folder):
        return

    print(f"\nProcessing {folder}...")
    
    for lang in LANGUAGES:
        for gender in ["male", "female"]:
            pattern = f"{lang}_{gender}_ai_*.wav"
            files = sorted(glob.glob(os.path.join(folder, pattern)))
            
            count = len(files)
            print(f"  {lang.upper()} {gender}: {count} files found")
            
            if count > target_per_gender:
                excess = count - target_per_gender
                print(f"    - Deleting {excess} excess files...")
                for f in files[target_per_gender:]:
                    os.remove(f)
            elif count < target_per_gender:
                print(f"    - WARNING: Missing {target_per_gender - count} files")

# Cleanup Train
cleanup_folder(train_dir, TRAIN_TARGET_PER_GENDER)

# Cleanup Test
cleanup_folder(test_dir, TEST_TARGET_PER_GENDER)

print("\n" + "=" * 60)
print("Cleanup Complete!")
print("=" * 60)
