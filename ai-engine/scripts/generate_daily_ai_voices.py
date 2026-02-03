"""
Generate AI voice samples for 5 languages: English, Hindi, Tamil, Malayalam, Telugu.
Uses Edge TTS and gTTS engines.
Generates flat directory structure compatible with dataset.py.
"""
import argparse
import csv
import os
import random
import shutil
import tempfile
import time
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

# Optional imports (handled gracefully if missing)
try:
    import edge_tts
    _EDGE_AVAILABLE = True
except Exception:
    edge_tts = None
    _EDGE_AVAILABLE = False

try:
    from gtts import gTTS
    _GTTS_AVAILABLE = True
except Exception:
    gTTS = None
    _GTTS_AVAILABLE = False

LANGUAGES = ["english", "hindi", "tamil", "malayalam", "telugu"]

# Edge TTS voices per language (free, no API key)
EDGE_VOICES = {
    "english": ["en-US-JennyNeural", "en-US-GuyNeural", "en-IN-NeerjaNeural", "en-IN-PrabhatNeural"],
    "hindi": ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"],
    "tamil": ["ta-IN-PallaviNeural", "ta-IN-ValluvarNeural"],
    "malayalam": ["ml-IN-SobhanaNeural", "ml-IN-MidhunNeural"],
    "telugu": ["te-IN-ShrutiNeural", "te-IN-MohanNeural"],
}

# gTTS language codes (free, no API key, all languages supported)
GTTS_LANG_CODES = {
    "english": "en",
    "hindi": "hi",
    "tamil": "ta",
    "malayalam": "ml",
    "telugu": "te",
}

NAMES = {
    "english": ["Aarav", "Neha", "Rahul", "Priya", "Karan", "Meera", "Vikram", "Riya"],
    "hindi": ["आरव", "नेहा", "राहुल", "प्रिय", "करण", "मीरा"],
    "tamil": ["ஆரவ்", "நேஹா", "ராகுல்", "பிரியா", "கரன்", "மீரா"],
    "malayalam": ["ആരവ്", "നേഹ", "രാഹുൽ", "പ്രിയ", "കരൺ", "മീര"],
    "telugu": ["ఆరవ్", "నేహా", "రాహుల్", "ప్రియా", "కరణ్", "మీरा"],
}

PLACES = {
    "english": ["office", "home", "market", "station", "hospital", "bank", "school", "park"],
    "hindi": ["ऑफिस", "घर", "बाज़ार", "स्टेशन", "अस्पताल", "बैंक", "स्कूल", "पार्क"],
    "tamil": ["அலுவலகம்", "வீடு", "சந்தை", "நிலையம்", "மருத்துவமனை", "வங்கி", "பள்ளி", "பூங்கா"],
    "malayalam": ["ഓഫീസ്", "വീട്", "മാർക്കറ്റ്", "സ്റ്റേഷൻ", "ആശുപത്രി", "ബാങ്ക്", "സ്കൂൾ", "പാർക്ക്"],
    "telugu": ["ఆఫీస్", "ఇల్లు", "మార్కెట్", "స్టేషన్", "ఆసుపత్రి", "బ్యాంక్", "పాఠశాల", "పార్క్"],
}

TIMES = {
    "english": ["today", "tomorrow", "this evening", "in the morning", "after lunch", "tonight"],
    "hindi": ["आज", "कल", "शाम को", "सुबह", "दोपहर के बाद", "रात को"],
    "tamil": ["இன்று", "நாளை", "மாலை", "காலை", "மதிய பிறகு", "இரவு"],
    "malayalam": ["ഇന്ന്", "നാളെ", "വൈകിട്ട്", "രാവിലെ", "ഉച്ചയ്ക്ക് ശേഷം", "രാത്രി"],
    "telugu": ["ఈ రోజు", "రేపు", "సాయంత్రం", "ఉదయం", "మధ్యాహ్నం తరువాత", "రాత్రి"],
}

EXPRESSIONS = {
    "english": ["I'm happy", "I'm a bit worried", "I'm excited", "I'm frustrated", "I feel relieved"],
    "hindi": ["मैं खुश हूँ", "मैं थोड़ा चिंतित हूँ", "मैं उत्साहित हूँ", "मैं परेशान हूँ", "मुझे राहत मिली"],
    "tamil": ["நான் மகிழ்ச்சியாய் இருக்கிறேன்", "நான் கொஞ்சம் கவலையாக இருக்கிறேன்", "நான் உற்சாகமாக இருக்கிறேன்", "நான் மன அழுத்தமாக இருக்கிறேன்", "நான் நிம்மதியாக இருக்கிறேன்"],
    "malayalam": ["ഞാന്‍ സന്തോഷത്തിലാണ്", "ഞാന്‍ അല്പം ആശങ്കയിലാണ്", "ഞാന്‍ ആവേശത്തിലാണ്", "ഞാന്‍ നിരാശയിലാണ്", "എനിക്ക് ആശ്വാസം തോന്നുന്നു"],
    "telugu": ["నేను సంతోషంగా ఉన్నాను", "నేను కొంచెం ఆందోళనలో ఉన్నాను", "నేను ఉత్సాహంగా ఉన్నాను", "నేను నిరాశగా ఉన్నాను", "నాకు ఊరట కలిగింది"],
}

TEMPLATES = {
    "english": [
        "{name}, {time} I will be at the {place}. Can you call me back?",
        "I just finished my work and I am heading home. {expression}.",
        "Please remind me about the meeting at the {place} {time}.",
        "I need to pick up some items from the {place} {time}.",
        "{expression}. I will explain everything once we meet.",
        "Let's catch up {time} for a quick coffee near the {place}.",
        "The traffic is heavy today, I might be late by ten minutes.",
        "Can you check if the package arrived at the {place}?",
        "I spoke to {name} and we agreed to meet {time}.",
    ],
    "hindi": [
        "{name}, {time} मैं {place} पर रहूँगा। बाद में कॉल करना।",
        "मैं काम खत्म करके घर जा रहा हूँ। {expression}।",
        "{time} {place} में मीटिंग याद दिलाना।",
        "मुझे {time} {place} से कुछ सामान लेना है।",
        "{expression}। मिलने पर सब बता दूँगा।",
        "चलो {time} {place} के पास चाय पीते हैं।",
        "आज ट्रैफिक ज्यादा है, मैं थोड़ी देर से पहुँचूँगा।",
        "{place} पर पैकेज आया है क्या, जरा देखना।",
        "मैंने {name} से बात की, हम {time} मिलेंगे।",
    ],
    "tamil": [
        "{name}, {time} நான் {place}க்கு வர்றேன். பிறகு பேசலாம்.",
        "வேலை முடிச்சு வீட்டுக்கு வர்றேன். {expression}.",
        "{time} {place}லில் சந்திப்பு இருக்கு, நினைவூட்டு.",
        "{time} {place}ல இருந்து சில பொருட்கள் வாங்கணும்.",
        "{expression}. சந்திச்சதும் எல்லாம் சொல்றேன்.",
        "{time} {place}க்குப் பக்கத்தில் ஒரு காபி சாப்பிடலாமா?",
        "இன்று போக்குவரத்து அதிகம், கொஞ்சம் தாமதமா வருவேன்.",
        "{place}க்கு வந்த பாக்கெட் வந்திருக்கு என்ன சொல்றாங்க?",
        "{name} கூட பேசினேன், {time} சந்திக்கலாம்."
    ],
    "malayalam": [
        "{name}, {time} ഞാൻ {place}ലേക്ക് പോകുന്നു. പിന്നീട് വിളിക്കൂ.",
        "ജോലി കഴിഞ്ഞ് വീട്ടിലേക്കാണ് പോകുന്നത്. {expression}.",
        "{time} {place}ലുള്ള മീറ്റിംഗ് ഓർമ്മിപ്പിക്കൂ.",
        "{time} {place}ൽ നിന്ന് ചില സാധനങ്ങൾ വാങ്ങണം.",
        "{expression}. കണ്ടാൽ എല്ലാം പറയാം.",
        "{time} {place}ക്ക് സമീപം ഒരു കാപ്പി കുടിക്കാമോ?",
        "ഇന്ന് ട്രാഫിക് കൂടുതലാണ്, കുറച്ച് വൈകും.",
        "{place}ലേക്ക് വന്ന പാക്കറ്റ് എത്തിയോ എന്ന് നോക്കൂ.",
        "{name}നോട് സംസാരിച്ചു, {time} കണ്ടുമുട്ടാം."
    ],
    "telugu": [
        "{name}, {time} నేను {place}కి వస్తున్నాను. తర్వాత మాట్లాడుదాం.",
        "పని అయిపోయి ఇంటికి వస్తున్నాను. {expression}.",
        "{time} {place}లో మీటింగ్ ఉంది, గుర్తు చేయండి.",
        "{time} {place} నుండి కొన్ని వస్తువులు తీసుకోవాలి.",
        "{expression}. కలిస్తే అన్నీ చెప్తాను.",
        "{time} {place} దగ్గర ఒక కాఫీ తాగుదామా?",
        "ఈ రోజు ట్రాఫిక్ ఎక్కువ, కొంచెం లేట్ అవుతాను.",
        "{place}కి వచ్చిన పార్సెల్ వచ్చిందా చెక్ చేయండి.",
        "{name}తో మాట్లాడాను, {time} కలుద్దాం."
    ],
}


@dataclass
class SampleMeta:
    file_path: str
    language: str
    tts_engine: str
    voice: str
    rate: str
    pitch: str
    text: str
    duration_sec: float
    split: str


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _wav_duration(path: str) -> float:
    return float(librosa.get_duration(path=path))


def _random_text(language: str, target_words: int) -> str:
    template = random.choice(TEMPLATES[language])
    text = template.format(
        name=random.choice(NAMES[language]),
        place=random.choice(PLACES[language]),
        time=random.choice(TIMES[language]),
        expression=random.choice(EXPRESSIONS[language]),
    )

    # Expand to approximate target length with varied fillers
    fillers = {
        "english": [
            "I just wanted to mention that.",
            "Let me know if that works for you.",
            "Hope everything is going well on your end.",
            "Just thought I should let you know about this.",
            "Please confirm when you get a chance.",
            "Looking forward to hearing back from you.",
            "This is quite important actually.",
            "I hope this message finds you well.",
            "Talk to you soon about this matter.",
        ],
        "hindi": [
            "मुझे यह बताना जरूरी था।",
            "कृपया मुझे बताएं कि यह ठीक है।",
            "उम्मीद है आप सब ठीक होंगे।",
            "बस आपको सूचित करना चाहता था।",
            "कृपया जल्दी जवाब दें।",
            "आपके उत्तर की प्रतीक्षा में।",
            "यह वाकई महत्वपूर्ण है।",
            "आशा है सब कुशल मंगल है।",
            "जल्द ही बात करते हैं।",
        ],
        "tamil": [
            "இதை உங்களுக்குத் தெரிவிக்க வேண்டும்.",
            "இது சரியா என்று சொல்லுங்கள்.",
            "எல்லாம் நலமாக இருக்கும் என்று நம்புகிறேன்.",
            "இதை உங்களுக்குத் தெரியப்படுத்த வேண்டும்.",
            "முடிந்தால் பதில் சொல்லுங்கள்.",
            "உங்கள் பதிலுக்காகக் காத்திருக்கிறேன்.",
            "இது மிகவும் முக்கியம் உண்மையில்.",
            "நீங்கள் நலமாக இருக்கிறீர்கள் என்று நம்புகிறேன்.",
            "விரைவில் பேசுவோம் இதைப் பற்றி.",
        ],
        "malayalam": [
            "ഇത് അറിയിക്കേണ്ടതായിരുന്നു.",
            "ഇത് ശരിയാണോ എന്ന് പറയൂ.",
            "എല്ലാം നന്നായിരിക്കുമെന്ന് പ്രതീക്ഷിക്കുന്നു.",
            "ഇത് നിങ്ങളെ അറിയിക്കണം.",
            "സാധിച്ചാൽ മറുപടി തരൂ.",
            "നിങ്ങളുടെ മറുപടിക്കായി കാത്തിരിക്കുന്നു.",
            "ഇത് വളരെ പ്രധാനമാണ് സത്യത്തിൽ.",
            "നിങ്ങൾ സുഖമായിരിക്കുമെന്ന് പ്രതീക്ഷിക്കുന്നു.",
            "ഉടൻ സംസാരിക്കാം ഇതിനെക്കുറിച്ച്.",
        ],
        "telugu": [
            "ఇది మీకు తెలియజేయాలి.",
            "ఇది సరైందేనా చెప్పండి.",
            "అన్నీ బాగానే ఉన్నాయని ఆశిస్తున్నాను.",
            "ఇది మీకు తెలియజేయాలనుకున్నాను.",
            "వీలైతే సమాధానం చెప్పండి.",
            "మీ సమాధానం కోసం ఎదురు చూస్తున్నాను.",
            "ఇది చాలా ముఖ్యమైనది నిజంగా.",
            "మీరు బాగున్నారని ఆశిస్తున్నాను.",
            "త్వరలో మాట్లాడుదాం దీని గురించి.",
        ],
    }[language]

    words = text.split()
    while len(words) < target_words:
        text = f"{text} {random.choice(fillers)}"
        words = text.split()
    return text


def _build_text_pool(language: str, pool_size: int, min_sec: float, max_sec: float) -> List[str]:
    """Step 1: Build multilingual daily-talk text pool with high variation."""
    pool: List[str] = []
    seen = set()
    attempts = 0
    max_attempts = pool_size * 50  # Increased for better uniqueness

    while len(pool) < pool_size and attempts < max_attempts:
        target_words = _target_word_count(min_sec, max_sec)
        text = _random_text(language, target_words)
        attempts += 1
        if text in seen:
            continue
        seen.add(text)
        pool.append(text)

    if len(pool) < pool_size:
        print(f"Warning: Could not build enough unique texts for {language}. Requested {pool_size}, got {len(pool)}")
    
    # Shuffle to ensure randomness
    random.shuffle(pool)
    return pool


def _build_multilingual_text_pools(
    total_per_lang: int,
    min_sec: float,
    max_sec: float,
) -> Dict[str, List[str]]:
    pools: Dict[str, List[str]] = {}
    # Build 2x text pool to maximize variation
    pool_size = max(1200, total_per_lang * 2)
    for language in LANGUAGES:
        pools[language] = _build_text_pool(language, pool_size, min_sec, max_sec)
    return pools


def _target_word_count(min_sec: float, max_sec: float) -> int:
    # Conservative estimate: 2.2 words/sec for mixed languages
    target_sec = random.uniform(min_sec, max_sec)
    return max(8, int(target_sec * 2.2))


class EdgeTTSEngine:
    name = "edge"

    def __init__(self):
        if not _EDGE_AVAILABLE:
            raise RuntimeError("edge-tts not installed")

    async def synthesize(self, text: str, out_path: str, voice: str, rate: str, pitch: str) -> None:
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
            pitch=pitch,
        )
        await communicate.save(out_path)


class GTTSEngine:
    name = "gtts"

    def __init__(self):
        if not _GTTS_AVAILABLE:
            raise RuntimeError("gTTS not installed. Run: pip install gtts")

    def synthesize(self, text: str, out_path: str, language: str, rate: str, pitch: str) -> str:
        lang_code = GTTS_LANG_CODES[language]
        
        # gTTS doesn't support rate/pitch, but we can simulate variation via speed parameter
        slow = False
        if "-" in rate:  # slower rate
            slow = True
        
        tts = gTTS(text=text, lang=lang_code, slow=slow)
        
        # Save to temporary MP3 first (gTTS outputs MP3)
        tmp_mp3 = out_path.replace(".wav", ".mp3")
        try:
            tts.save(tmp_mp3)
            
            # Convert MP3 to WAV using librosa
            # Note: librosa.load might be slow. Optimization: use ffmpeg directly if possible,
            # but sticking to librosa for compatibility since we know it's installed.
            audio, sr = librosa.load(tmp_mp3, sr=16000)
            sf.write(out_path, audio, sr)
        finally:
            # Clean up temp MP3
            if os.path.exists(tmp_mp3):
                os.remove(tmp_mp3)
        
        return f"gtts-{lang_code}"


def _pick_rate() -> str:
    return random.choice(["-10%", "-5%", "+1%", "+5%", "+10%"])


def _pick_pitch() -> str:
    return random.choice(["-20Hz", "-10Hz", "+1Hz", "+10Hz", "+20Hz"])


def _generate_one(
    language: str,
    engine_name: str,
    engine,
    out_dir: str,
    min_sec: float,
    max_sec: float,
    text_pool: List[str],
    max_attempts: int = 40,
) -> SampleMeta:
    durations_tried = []
    for attempt in range(max_attempts):
        # Step 2: Select from the pre-built text pool
        if len(text_pool) == 1:
            text = text_pool[0]
            if attempt > 0:
                extra_words = 6 * (attempt // 5)
                text = _random_text(language, _target_word_count(min_sec, max_sec) + extra_words)
        else:
            text = random.choice(text_pool)
        rate = _pick_rate()
        pitch = _pick_pitch()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, f"tmp_{int(time.time() * 1000)}.wav")
            voice = "default"

            try:
                if engine_name == "edge":
                    voice = random.choice(EDGE_VOICES[language])
                    asyncio.run(engine.synthesize(text, tmp_path, voice, rate, pitch))
                elif engine_name == "gtts":
                    voice = engine.synthesize(text, tmp_path, language, rate, pitch)
                else:
                    raise RuntimeError(f"Unknown engine: {engine_name}")

                # Step 3: Enforce 5–20 sec duration (or as configured)
                duration = _wav_duration(tmp_path)
                durations_tried.append(duration)
                
                if min_sec <= duration <= max_sec:
                    _ensure_dir(out_dir)
                    # Use language in filename to distinguish
                    filename = f"{engine_name}_{language}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}.wav"
                    final_path = os.path.join(out_dir, filename)
                    shutil.copyfile(tmp_path, final_path)
                    return SampleMeta(
                        file_path=final_path,
                        language=language,
                        tts_engine=engine_name,
                        voice=voice,
                        rate=rate,
                        pitch=pitch,
                        text=text,
                        duration_sec=duration,
                        split=os.path.basename(os.path.dirname(out_dir)), # This will be 'ai'
                    )
            except Exception as e:
                # print(f"Generation error: {e}")
                pass

    avg_dur = sum(durations_tried) / len(durations_tried) if durations_tried else 0
    print(f"Warning: Failed after {max_attempts} attempts. Durations: min={min(durations_tried) if durations_tried else 0:.1f}s, max={max(durations_tried) if durations_tried else 0:.1f}s, avg={avg_dur:.1f}s")
    raise RuntimeError(f"Failed to generate clip in {max_attempts} attempts for {engine_name}:{language}")


def _init_engines() -> Dict[str, object]:
    engines: Dict[str, object] = {}
    if _EDGE_AVAILABLE:
        engines["edge"] = EdgeTTSEngine()
    if _GTTS_AVAILABLE:
        engines["gtts"] = GTTSEngine()
    return engines


def generate_dataset_custom(
    base_dir: str,
    engine_configs: Dict[str, Tuple[int, int]],
    min_sec: float,
    max_sec: float,
    seed: int,
    metadata_path: str,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    # Step 1: Build multilingual daily-talk text pools
    total_per_lang = sum(train + test for train, test in engine_configs.values())
    print(f"\nBuilding text pools for {len(LANGUAGES)} languages...")
    text_pools = _build_multilingual_text_pools(total_per_lang, min_sec, max_sec)
    print(f"✓ Text pools built successfully\n")

    engines = _init_engines()
    if not engines:
        print("Error: No TTS engines available!")
        return

    records: List[SampleMeta] = []

    # Flattened directory structure: data/train/ai/ and data/test/ai/
    train_dir = os.path.join(base_dir, "train", "ai")
    test_dir = os.path.join(base_dir, "test", "ai")
    _ensure_dir(train_dir)
    _ensure_dir(test_dir)

    for language in LANGUAGES:
        print(f"\n=== Generating for {language} ===")

        # Create a copy of the text pool and pop unique texts for each sample
        available_texts = text_pools[language].copy()
        random.shuffle(available_texts)  # Shuffle for randomness

        for engine_name, (train_count, test_count) in engine_configs.items():
            if engine_name not in engines:
                print(f"Skipping {engine_name} (not available)")
                continue

            engine = engines[engine_name]
            print(f"  -> {engine_name}: {train_count} train + {test_count} test")

            for i in range(train_count):
                # Pop unique text from pool
                if available_texts:
                    unique_text_pool = [available_texts.pop()]
                else:
                    unique_text_pool = [_random_text(language, _target_word_count(min_sec, max_sec))]
                
                try:
                    meta = _generate_one(language, engine_name, engine, train_dir, min_sec, max_sec, unique_text_pool)
                    # Update metadata path to be relative or consistent
                    records.append(meta)
                    if (i + 1) % 10 == 0:
                        print(f"     {engine_name} train: {i + 1}/{train_count}")
                except Exception as e:
                    print(f"Error generating sample: {e}")

            for i in range(test_count):
                # Pop unique text from pool
                if available_texts:
                    unique_text_pool = [available_texts.pop()]
                else:
                    unique_text_pool = [_random_text(language, _target_word_count(min_sec, max_sec))]
                
                try:
                    meta = _generate_one(language, engine_name, engine, test_dir, min_sec, max_sec, unique_text_pool)
                    # Manually set split to 'test' since _generate_one might derive it from path
                    meta.split = "test" 
                    records.append(meta)
                    if (i + 1) % 10 == 0:
                        print(f"     {engine_name} test: {i + 1}/{test_count}")
                except Exception as e:
                    print(f"Error generating sample: {e}")

    # Store metadata
    _ensure_dir(os.path.dirname(metadata_path))
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file_path",
            "language",
            "tts_engine",
            "voice",
            "rate",
            "pitch",
            "text",
            "duration_sec",
            "split",
        ])
        for r in records:
            writer.writerow([
                r.file_path,
                r.language,
                r.tts_engine,
                r.voice,
                r.rate,
                r.pitch,
                r.text,
                f"{r.duration_sec:.3f}",
                r.split,
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate daily-talk AI voice dataset from scratch")
    parser.add_argument("--base-dir", default=os.path.join(os.path.dirname(__file__), "..", "data"))
    parser.add_argument("--edge-count", type=int, default=125, help="Edge TTS train samples per language")
    parser.add_argument("--gtts-count", type=int, default=75, help="gTTS train samples per language")
    parser.add_argument("--edge-test", type=int, default=35, help="Edge TTS test samples per language")
    parser.add_argument("--gtts-test", type=int, default=15, help="gTTS test samples per language")
    parser.add_argument("--min-sec", type=float, default=2.0)
    parser.add_argument("--max-sec", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata", default=os.path.join(os.path.dirname(__file__), "..", "data", "ai_generated_metadata.csv"))
    args = parser.parse_args()

    # Total numbers
    train_per_lang = args.edge_count + args.gtts_count
    test_per_lang = args.edge_test + args.gtts_test
    total_per_lang = train_per_lang + test_per_lang
    
    print("=" * 70)
    print("AI VOICE DATASET GENERATOR (MULTILINGUAL)")
    print("=" * 70)
    print(f"Languages: {', '.join(LANGUAGES)}")
    print(f"Target: {total_per_lang} samples per language (Total: {total_per_lang * len(LANGUAGES)})")
    print(f"Duration: {args.min_sec}-{args.max_sec} seconds per clip")
    print(f"Split: {train_per_lang} train / {test_per_lang} test per language")
    print(f"  Edge: {args.edge_count} train + {args.edge_test} test")
    print(f"  gTTS: {args.gtts_count} train + {args.gtts_test} test")
    print("=" * 70)

    # Configs
    engine_configs = {
        "edge": (args.edge_count, args.edge_test),
        "gtts": (args.gtts_count, args.gtts_test),
    }

    generate_dataset_custom(
        base_dir=os.path.abspath(args.base_dir),
        engine_configs=engine_configs,
        min_sec=args.min_sec,
        max_sec=args.max_sec,
        seed=args.seed,
        metadata_path=os.path.abspath(args.metadata),
    )
    
    print("\n" + "=" * 70)
    print("✓ GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
