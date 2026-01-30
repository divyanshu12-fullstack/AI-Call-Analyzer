import os
import librosa
import soundfile as sf

TARGET_SR = 16000
DURATION = 5  # seconds

def preprocess_audio(input_path, output_path):
    audio, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)

    max_length = TARGET_SR * DURATION

    if len(audio) > max_length:
        audio = audio[:max_length]
    else:
        padding = max_length - len(audio)
        audio = librosa.util.pad_center(audio, padding)

    sf.write(output_path, audio, TARGET_SR)

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.endswith(".wav"):
            preprocess_audio(
                os.path.join(input_folder, file),
                os.path.join(output_folder, file)
            )

if __name__ == "__main__":
    process_folder("raw_data/human", "data/human")
    process_folder("raw_data/ai", "data/ai")
    print("Preprocessing complete")
