
import os
import glob
from inference import VoiceDetector

def test_new_voices():
    # Path to the newly generated validation data
    base_dir = "data_validation/test/ai"
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist.")
        return

    print("Loading VoiceDetector...")
    detector = VoiceDetector()

    # Metrics
    total_files = 0
    detected_ai = 0
    results_by_lang = {}

    # Get all .wav files
    wav_files = glob.glob(os.path.join(base_dir, "*.wav"))
    
    if not wav_files:
        print("No .wav files found in validation directory.")
        return

    print(f"\nStarting validation on {len(wav_files)} new AI files...\n")
    print(f"{'Filename':<40} | {'Lang':<10} | {'Prediction':<15} | {'Conf':<8}")
    print("-" * 80)

    for file_path in wav_files:
        filename = os.path.basename(file_path)
        
        # Try to infer language from filename (format: engine_lang_time_rand.wav)
        # e.g., edge_english_123_456.wav
        parts = filename.split('_')
        if len(parts) >= 2:
            language = parts[1]
        else:
            language = "unknown"

        if language not in results_by_lang:
            results_by_lang[language] = {"total": 0, "correct": 0}

        try:
            result = detector.predict_from_file(file_path)
            is_ai = result["classification"] == "AI_GENERATED"
            confidence = result["confidence"]
            
            total_files += 1
            results_by_lang[language]["total"] += 1

            if is_ai:
                detected_ai += 1
                results_by_lang[language]["correct"] += 1
                status = "✅ AI"
            else:
                status = "❌ HUMAN"

            print(f"{filename[:38]:<40} | {language:<10} | {status:<15} | {confidence:.2%}")

        except Exception as e:
            print(f"{filename[:38]:<40} | {language:<10} | ERROR: {str(e)}")

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy: {detected_ai}/{total_files} ({detected_ai/total_files*100:.2f}%)")
    print("-" * 60)
    for lang, stats in results_by_lang.items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"] * 100
            print(f"{lang.capitalize():<12}: {stats['correct']}/{stats['total']} ({acc:.2f}%)")
    print("=" * 60)

if __name__ == "__main__":
    test_new_voices()
