#!/usr/bin/env python3
"""Generate Tamil, Malayalam, Telugu voices using Edge TTS."""
import asyncio
import csv
import os
import random
from datetime import datetime

try:
    import edge_tts
except ImportError:
    print("ERROR: edge-tts not installed. Run: pip install edge-tts")
    exit(1)

# Configuration
LANGUAGES = {
    "tamil": ("ta", "ta-IN-PallaviNeural"),
    "malayalam": ("ml", "ml-IN-SobhanaNeural"),
    "telugu": ("te", "te-IN-ShrutiNeural"),
}

TEXTS = {
    "tamil": [
        "உங்கள் கணக்கில் சந்தேகமான நடவடிக்கை கண்டறியப்பட்டது.",
        "உங்கள் கணக்கு தற்காலிகமாக முடக்கப்படும்; தயவுசெய்து உறுதிப்படுத்தவும்.",
        "உங்கள் அடையாளத்தைச் சரிபார்க்க வேண்டும்.",
        "உங்கள் KYC இன்று புதுப்பிக்கப்பட வேண்டும்.",
        "உங்கள் கார்டு பாதுகாப்புக்காக தடை செய்யப்பட்டுள்ளது.",
    ],
    "malayalam": [
        "നിങ്ങളുടെ അക്കൗണ്ടിൽ സംശയാസ്പദ പ്രവർത്തനം കണ്ടെത്തി.",
        "നിങ്ങളുടെ അക്കൗണ്ട് താൽക്കാലികമായി സസ്പെൻഡ് ചെയ്യപ്പെടും; സ്ഥിരീകരിക്കുക.",
        "നിങ്ങളുടെ തിരിച്ചറിയൽ സ്ഥിരീകരണം ആവശ്യമാണ്.",
        "നിങ്ങളുടെ KYC ഇന്ന് പുതുക്കണം.",
        "സുരക്ഷാ കാരണങ്ങളാൽ നിങ്ങളുടെ കാർഡ് ബ്ലോക്ക് ചെയ്തിട്ടുണ്ട്.",
    ],
    "telugu": [
        "మీ ఖాతాలో అనుమానాస్పద కార్యకలాపం గుర్తించబడింది.",
        "మీ ఖాతాను తాత్కాలికంగా నిలిపివేస్తారు; దయచేసి నిర్ధారించండి.",
        "మీ గుర్తింపును ధృవీకరించాలి.",
        "మీ KYC ఈ రోజే అప్డేట్ చేయాలి.",
        "భద్రత కారణాల వల్ల మీ కార్డ్ బ్లాక్ చేయబడింది.",
    ],
}

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train", "ai")
TEST_DIR = os.path.join(BASE_DIR, "data", "test", "ai")
METADATA_PATH = os.path.join(BASE_DIR, "data", "ai_generated_metadata.csv")

TRAIN_COUNT = 200
TEST_COUNT = 50
SEED = 42


async def generate_tts(text: str, voice: str, filepath: str, max_retries: int = 3) -> bool:
    """Generate TTS audio using Edge TTS with retry logic."""
    for attempt in range(max_retries):
        try:
            communicate = edge_tts.Communicate(text=text, voice=voice)
            await communicate.save(filepath)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"      ERROR (attempt {attempt+1}): {type(e).__name__}. Retrying...")
                await asyncio.sleep(2)  # Wait before retry
            else:
                print(f"      ERROR (final): {e}")
                return False


async def generate_split(split: str, count: int) -> list[dict]:
    """Generate audio for a split (train/test)."""
    metadata_rows = []
    out_dir = TRAIN_DIR if split == "train" else TEST_DIR

    for lang_name, (lang_code, voice) in LANGUAGES.items():
        lang_dir = os.path.join(out_dir, lang_name)
        os.makedirs(lang_dir, exist_ok=True)

        print(f"\nGenerating {count} samples for {lang_name} ({split})...")
        texts = TEXTS[lang_name]
        generated_count = 0

        for idx in range(count):
            text = random.choice(texts)
            filename = f"{lang_name}_ai_{split}_{idx + 1:04d}.wav"
            filepath = os.path.join(lang_dir, filename)

            success = await generate_tts(text, voice, filepath)
            if success:
                generated_count += 1
                metadata_rows.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "split": split,
                    "language": lang_name,
                    "text": text,
                    "path": filepath,
                    "speaker_wav": "",
                    "speed": 1.0,
                    "model": "edge-tts",
                    "device": "cloud",
                })

            if (idx + 1) % 50 == 0:
                print(f"  Generated {generated_count}/{idx + 1} for {lang_name} ({split})")

        print(f"  Total: {generated_count}/{count} for {lang_name} ({split})")

    return metadata_rows


async def main():
    """Main generation pipeline."""
    random.seed(SEED)
    print(f"Starting Tamil/Malayalam/Telugu voice generation...")
    print(f"Train samples per language: {TRAIN_COUNT}")
    print(f"Test samples per language: {TEST_COUNT}")

    all_metadata = []

    # Generate training data
    train_metadata = await generate_split("train", TRAIN_COUNT)
    all_metadata.extend(train_metadata)

    # Generate test data
    test_metadata = await generate_split("test", TEST_COUNT)
    all_metadata.extend(test_metadata)

    # Write metadata
    if all_metadata:
        is_new = not os.path.exists(METADATA_PATH)
        with open(METADATA_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "split", "language", "text", "path",
                "speaker_wav", "speed", "model", "device"
            ])
            if is_new:
                writer.writeheader()
            writer.writerows(all_metadata)
        print(f"\nMetadata written to: {METADATA_PATH}")

    # Summary
    print("\n=== GENERATION SUMMARY ===")
    for split, base_dir in (("train", TRAIN_DIR), ("test", TEST_DIR)):
        print(f"\n{split.upper()}:")
        for lang_name in LANGUAGES.keys():
            lang_dir = os.path.join(base_dir, lang_name)
            if os.path.isdir(lang_dir):
                count = sum(1 for f in os.listdir(lang_dir) if f.endswith(".wav"))
                print(f"  {lang_name}: {count} files")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
