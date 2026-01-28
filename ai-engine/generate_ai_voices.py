"""
Generate AI voice samples using Google Text-to-Speech (gTTS).
These will serve as the "AI-generated" class for our detector.
"""
import os
from gtts import gTTS
import librosa
import soundfile as sf
import random

# Configuration
TRAIN_COUNT = 100
TEST_COUNT = 20
TARGET_SR = 16000

# Sample texts to generate speech from (varied lengths and content)
SAMPLE_TEXTS = [
    "Hello, how are you doing today?",
    "The weather is really nice outside.",
    "I would like to schedule a meeting for tomorrow.",
    "Please confirm your account details.",
    "Your order has been shipped and will arrive soon.",
    "Thank you for calling our customer service.",
    "Press one for sales, press two for support.",
    "Your balance is available on the mobile app.",
    "We are experiencing higher than normal call volumes.",
    "Please hold while we transfer your call.",
    "This call may be recorded for quality purposes.",
    "Enter your pin number followed by the pound key.",
    "Your appointment has been confirmed.",
    "The office will be closed on Monday.",
    "Please leave a message after the tone.",
    "Your prescription is ready for pickup.",
    "The estimated wait time is five minutes.",
    "Thank you for your patience.",
    "Have a great day and goodbye.",
    "Please speak clearly after the beep.",
    "Your flight has been delayed by two hours.",
    "The reservation is confirmed for next Friday.",
    "We value your feedback and suggestions.",
    "The system will restart in ten minutes.",
    "Your password has been reset successfully.",
    "Welcome to our automated system.",
    "For English, press one. For Spanish, press two.",
    "Your account has been credited.",
    "Please verify your identity.",
    "The transaction was completed successfully.",
]

# Languages to use (varying the TTS voice)
LANGUAGES = ["en", "en-au", "en-uk", "en-us", "en-in"]

# Create folders
train_ai_dir = "data/train/ai"
test_ai_dir = "data/test/ai"
temp_dir = "temp_tts"

os.makedirs(train_ai_dir, exist_ok=True)
os.makedirs(test_ai_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

print("Generating AI voice samples using gTTS...")

count = 0
total_needed = TRAIN_COUNT + TEST_COUNT

while count < total_needed:
    # Pick random text and language
    text = random.choice(SAMPLE_TEXTS)
    lang = random.choice(LANGUAGES)
    
    try:
        # Generate TTS
        tts = gTTS(text=text, lang=lang.split("-")[0], tld=lang.split("-")[1] if "-" in lang else "com")
        
        # Save to temp mp3
        temp_mp3 = os.path.join(temp_dir, f"temp_{count}.mp3")
        tts.save(temp_mp3)
        
        # Load and convert to 16kHz mono WAV
        audio, sr = librosa.load(temp_mp3, sr=TARGET_SR, mono=True)
        
        # Trim to max 5 seconds
        max_samples = TARGET_SR * 5
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # Skip very short clips
        if len(audio) < TARGET_SR:
            os.remove(temp_mp3)
            continue
        
        # Save to train or test folder
        if count < TRAIN_COUNT:
            out_path = os.path.join(train_ai_dir, f"ai_{count:04d}.wav")
            sf.write(out_path, audio, TARGET_SR)
            print(f"[Train] Generated {count + 1}/{TRAIN_COUNT}")
        else:
            test_idx = count - TRAIN_COUNT
            out_path = os.path.join(test_ai_dir, f"ai_{test_idx:04d}.wav")
            sf.write(out_path, audio, TARGET_SR)
            print(f"[Test] Generated {test_idx + 1}/{TEST_COUNT}")
        
        # Cleanup temp file
        os.remove(temp_mp3)
        count += 1
        
    except Exception as e:
        print(f"Error generating sample {count}: {e}")
        continue

print(f"\n✓ AI voice generation finished!")
print(f"  Train samples: {min(count, TRAIN_COUNT)}")
print(f"  Test samples: {max(0, count - TRAIN_COUNT)}")

# Cleanup temp directory
import shutil
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
