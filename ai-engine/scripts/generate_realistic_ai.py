
"""
Generate Realistic AI Voice Samples (English only for now).
Uses edge-tts for neural voices and adds noise/telephony effects.
"""
import asyncio
import os
import random
import edge_tts
import librosa
import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter
from pydub import AudioSegment
import io

# Configuration
TRAIN_COUNT = 150  # Increased since we have more variety
TEST_COUNT = 30
TARGET_SR = 16000

# Edge-TTS Voices (English)
# These are high quality neural voices
VOICES = [
    "en-US-AriaNeural",
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-AU-NatashaNeural",
    "en-CA-LiamNeural",
    "en-IE-EmilyNeural"
]

SAMPLE_TEXTS = [
    "Hello, I am calling from Amazon regarding your recent order.",
    "This is a final notice about your car's extended warranty.",
    "Hi, this is Sarah from the recruitment team. Are you available for a quick chat?",
    "We noticed some unusual activity on your credit card ending in 4242.",
    "Good morning. I'm calling to confirm your appointment for next Tuesday.",
    "Congratulations! You have been selected for a special holiday offer.",
    "Please do not hang up. This is an important message from your bank.",
    "Hello, can you hear me? I think we have a bad connection.",
    "I'm calling to conduct a brief survey about your internet service.",
    "Your package was unable to be delivered. Please update your address.",
    "This call is being recorded for quality assurance purposes.",
    "Press one to speak with a representative.",
    "We successfully received your payment of $49.99.",
    "Your subscription will automatically renew tomorrow.",
    "I am an artificial intelligence assistant designed to help you.",
    "The verification code is 4 8 2 9.",
    "Thank you for your patience. All our agents are currently busy.",
    "Please stay on the line for the next available operator.",
    "Your medical test results are ready to be viewed online.",
    "We have a special promotion just for you.",
    "Is this the correct number for the homeowner?",
    "I'm sorry, I didn't catch that. Could you repeat it?",
    "This is a courtesy call to remind you of your upcoming payment.",
    "Your account has been temporarily locked due to security concerns.",
    "Hello? Is anyone there?",
    "We have a delivery for you requiring a signature.",
    "Please listen carefully as our menu options have changed.",
    "To opt out of future calls, please press nine.",
    "I'm calling on behalf of the local charity drive.",
    "Have you considered upgrading your mobile plan?",
]

# Output Directories
train_ai_dir = "data/train/ai"
test_ai_dir = "data/test/ai"
temp_dir = "temp_realistic"

os.makedirs(train_ai_dir, exist_ok=True)
os.makedirs(test_ai_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

# --- Augmentation Functions ---

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_telephony_filter(data, sr):
    """Simulate phone line bandwidth (300Hz - 3400Hz)."""
    lowcut = 300.0
    highcut = 3400.0
    b, a = butter_bandpass(lowcut, highcut, sr, order=6)
    y = lfilter(b, a, data)
    return y

def add_noise(data, noise_factor=0.005):
    """Add white noise."""
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data

async def generate_sample(text, voice, output_path, idx):
    temp_mp3 = os.path.join(temp_dir, f"raw_{idx}.mp3")
    
    try:
        # 1. Generate clean AI Audio
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(temp_mp3)
        
        # 2. Load with librosa
        y, sr = librosa.load(temp_mp3, sr=TARGET_SR, mono=True)
        
        # 3. Randomly apply augmentations
        
        # A. Telephony Filter (80% chance - since most scam calls are phones)
        if random.random() < 0.8:
            y = apply_telephony_filter(y, sr)
            
        # B. Background Noise (100% chance, varying levels)
        noise_level = random.uniform(0.001, 0.015) 
        y = add_noise(y, noise_factor=noise_level)
        
        # 4. Normalize
        y = y / np.max(np.abs(y))
        
        # 5. Length Check
        if len(y) < TARGET_SR: # Skip < 1s
            return False
            
        # Trim to 5s max
        max_len = TARGET_SR * 5
        if len(y) > max_len:
            y = y[:max_len]
            
        # 6. Save
        sf.write(output_path, y, sr)
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)

async def main():
    print(f"Generating realistic AI voices...")
    
    # --- Generate Train ---
    count = 0
    while count < TRAIN_COUNT:
        text = random.choice(SAMPLE_TEXTS)
        voice = random.choice(VOICES)
        fname = os.path.join(train_ai_dir, f"realistic_ai_{count:04d}.wav")
        
        if await generate_sample(text, voice, fname, count):
            count += 1
            if count % 10 == 0:
                print(f"[Train] {count}/{TRAIN_COUNT}")

    # --- Generate Test ---
    count = 0
    while count < TEST_COUNT:
        text = random.choice(SAMPLE_TEXTS)
        voice = random.choice(VOICES)
        fname = os.path.join(test_ai_dir, f"realistic_ai_{count:04d}.wav")
        
        if await generate_sample(text, voice, fname, count + 1000):
            count += 1
            if count % 5 == 0:
                print(f"[Test] {count}/{TEST_COUNT}")

    print("Done!")
    
    # Cleanup
    try:
        os.rmdir(temp_dir)
    except:
        pass

if __name__ == "__main__":
    loop = asyncio.get_event_loop_policy().get_event_loop()
    try:
        loop.run_until_complete(main())
    except RuntimeError:
         asyncio.run(main())
