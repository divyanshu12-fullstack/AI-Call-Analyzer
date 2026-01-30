"""
Generate multilingual AI voice samples using gTTS.
Supports Telugu, Malayalam, Hindi, Tamil, and English with both male and female voices.
"""
import os
from gtts import gTTS
import librosa
import soundfile as sf
import random
import shutil

# Configuration
LANGUAGES = {
    "te": {"name": "Telugu", "gtts_code": "te"},
    "ml": {"name": "Malayalam", "gtts_code": "ml"},
    "hi": {"name": "Hindi", "gtts_code": "hi"},
    "ta": {"name": "Tamil", "gtts_code": "ta"},
    "en": {"name": "English", "gtts_code": "en"}
}

TRAIN_PER_LANG = 100  # 50 male, 50 female
TEST_PER_LANG = 26    # 13 male, 13 female
TARGET_SR = 16000

# Sample texts for each language
SAMPLE_TEXTS = {
    "en": [
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
    ],
    "hi": [
        "नमस्ते, आप कैसे हैं?",
        "मौसम बहुत अच्छा है।",
        "कृपया अपना खाता विवरण की पुष्टि करें।",
        "आपका ऑर्डर भेज दिया गया है।",
        "हमारी ग्राहक सेवा को कॉल करने के लिए धन्यवाद।",
        "बिक्री के लिए एक दबाएं, समर्थन के लिए दो।",
        "आपका शेष मोबाइल ऐप पर उपलब्ध है।",
        "कृपया अपने पिन नंबर दर्ज करें।",
        "आपकी नियुक्ति की पुष्टि हो गई है।",
        "कार्यालय सोमवार को बंद रहेगा।",
        "कृपया टोन के बाद संदेश छोड़ें।",
        "आपका प्रिस्क्रिप्शन पिकअप के लिए तैयार है।",
        "अनुमानित प्रतीक्षा समय पांच मिनट है।",
        "आपके धैर्य के लिए धन्यवाद।",
        "आपका दिन शुभ हो और अलविदा।",
    ],
    "ta": [
        "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "வானிலை மிகவும் நன்றாக உள்ளது।",
        "உங்கள் கணக்கு விவரங்களை உறுதிப்படுத்தவும்.",
        "உங்கள் ஆர்டர் அனுப்பப்பட்டுள்ளது.",
        "எங்கள் வாடிக்கையாளர் சேவையை அழைத்ததற்கு நன்றி.",
        "விற்பனைக்கு ஒன்றை அழுத்தவும், ஆதரவுக்கு இரண்டு.",
        "உங்கள் இருப்பு மொபைல் பயன்பாட்டில் கிடைக்கும்.",
        "தயவுசெய்து உங்கள் பின் எண்ணை உள்ளிடவும்.",
        "உங்கள் சந்திப்பு உறுதிப்படுத்தப்பட்டது.",
        "அலுவலகம் திங்கட்கிழமை மூடப்படும்.",
        "தயவுசெய்து டோனுக்குப் பிறகு செய்தியை விடுங்கள்.",
        "உங்கள் மருந்துச்சீட்டு எடுப்பதற்கு தயாராக உள்ளது.",
        "மதிப்பிடப்பட்ட காத்திருப்பு நேரம் ஐந்து நிமிடங்கள்.",
        "உங்கள் பொறுமைக்கு நன்றி.",
        "நல்ல நாளாக இருக்கட்டும், விடைபெறுகிறேன்.",
    ],
    "te": [
        "హలో, మీరు ఎలా ఉన్నారు?",
        "వాతావరణం చాలా బాగుంది.",
        "దయచేసి మీ ఖాతా వివరాలను నిర్ధారించండి.",
        "మీ ఆర్డర్ పంపించబడింది.",
        "మా కస్టమర్ సేవకు కాల్ చేసినందుకు ధన్యవాదాలు.",
        "అమ్మకాల కోసం ఒకటి నొక్కండి, మద్దతు కోసం రెండు.",
        "మీ బ్యాలెన్స్ మొబైల్ యాప్‌లో అందుబాటులో ఉంది.",
        "దయచేసి మీ పిన్ నంబర్‌ను నమోదు చేయండి.",
        "మీ అపాయింట్‌మెంట్ నిర్ధారించబడింది.",
        "కార్యాలయం సోమవారం మూసివేయబడుతుంది.",
        "దయచేసి టోన్ తర్వాత సందేశాన్ని వదిలివేయండి.",
        "మీ ప్రిస్క్రిప్షన్ పికప్ కోసం సిద్ధంగా ఉంది.",
        "అంచనా వేచి ఉండే సమయం ఐదు నిమిషాలు.",
        "మీ సహనానికి ధన్యవాదాలు.",
        "మంచి రోజు గడపండి మరియు వీడ్కోలు.",
    ],
    "ml": [
        "ഹലോ, നിങ്ങൾക്ക് എങ്ങനെയുണ്ട്?",
        "കാലാവസ്ഥ വളരെ നല്ലതാണ്.",
        "ദയവായി നിങ്ങളുടെ അക്കൗണ്ട് വിശദാംശങ്ങൾ സ്ഥിരീകരിക്കുക.",
        "നിങ്ങളുടെ ഓർഡർ അയച്ചു.",
        "ഞങ്ങളുടെ ഉപഭോക്തൃ സേവനം വിളിച്ചതിന് നന്ദി.",
        "വിൽപ്പനയ്‌ക്കായി ഒന്ന് അമർത്തുക, പിന്തുണയ്‌ക്കായി രണ്ട്.",
        "നിങ്ങളുടെ ബാലൻസ് മൊബൈൽ ആപ്പിൽ ലഭ്യമാണ്.",
        "ദയവായി നിങ്ങളുടെ പിൻ നമ്പർ നൽകുക.",
        "നിങ്ങളുടെ അപ്പോയിന്റ്മെന്റ് സ്ഥിരീകരിച്ചു.",
        "ഓഫീസ് തിങ്കളാഴ്ച അടച്ചിരിക്കും.",
        "ദയവായി ടോണിന് ശേഷം സന്ദേശം ഇടുക.",
        "നിങ്ങളുടെ കുറിപ്പടി പിക്കപ്പിന് തയ്യാറാണ്.",
        "കണക്കാക്കിയ കാത്തിരിപ്പ് സമയം അഞ്ച് മിനിറ്റ്.",
        "നിങ്ങളുടെ ക്ഷമയ്ക്ക് നന്ദി.",
        "നല്ല ദിവസം കഴിയട്ടെ, വിട.",
    ]
}

# Voice variants with pitch shifting for male voices
VOICE_VARIANTS = {
    "male": {
        "tlds": ["co.uk", "com.au"],
        "pitch_shift": -4.0,
        "speed_factor": 0.95
    },
    "female": {
        "tlds": ["co.in", "ca", "ie", "com"],
        "pitch_shift": 0.0,
        "speed_factor": 1.05
    }
}

# Create folders
train_ai_dir = "data/train/ai"
test_ai_dir = "data/test/ai"
temp_dir = "temp_tts"

os.makedirs(train_ai_dir, exist_ok=True)
os.makedirs(test_ai_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

# Define generate_sample function BEFORE it's called
def generate_sample(lang_code, gender, text, output_dir, index):
    temp_mp3 = None
    try:
        variant = VOICE_VARIANTS[gender]
        tld = random.choice(variant["tlds"])
        
        # Generate TTS
        tts = gTTS(text=text, lang=LANGUAGES[lang_code]["gtts_code"], tld=tld, slow=False)
        temp_mp3 = os.path.join(temp_dir, f"temp_{lang_code}_{gender}_{random.randint(0,10000)}.mp3")
        tts.save(temp_mp3)
        
        # Load
        audio, sr = librosa.load(temp_mp3, sr=None, mono=True)
        
        # Pitch shift for male
        if variant["pitch_shift"] != 0:
            try:
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=variant["pitch_shift"])
            except:
                pass

        # Speed adjustment
        if variant["speed_factor"] != 1.0:
            audio = librosa.effects.time_stretch(audio, rate=variant["speed_factor"])
        
        # Resample
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
            
        # Ensure min length (1 sec)
        if len(audio) < TARGET_SR:
            os.remove(temp_mp3)
            return False
            
        # Max length 5 sec
        max_samples = TARGET_SR * 5
        if len(audio) > max_samples:
            audio = audio[:max_samples]
            
        # Save
        filename = f"{lang_code}_{gender}_ai_{index:04d}.wav"
        sf.write(os.path.join(output_dir, filename), audio, TARGET_SR)
        
        os.remove(temp_mp3)
        return True
    except Exception as e:
        if temp_mp3 and os.path.exists(temp_mp3):
            os.remove(temp_mp3)
        return False

def get_existing_counts(lang_code, gender, split_dir):
    files = [f for f in os.listdir(split_dir) if f.startswith(f"{lang_code}_{gender}_")]
    return len(files)

print("=" * 60)
print("Generating Multilingual AI Voice Samples")
print("=" * 60)
print(f"Languages: {', '.join([v['name'] for v in LANGUAGES.values()])}")
print(f"Target: {TRAIN_PER_LANG} train + {TEST_PER_LANG} test per language")
print(f"Structure: 50 Male / 50 Female per language (Train)")
print("=" * 60)

# Generate for each language
for lang_code, lang_info in LANGUAGES.items():
    lang_name = lang_info["name"]
    texts = SAMPLE_TEXTS.get(lang_code, SAMPLE_TEXTS["en"])
    
    print(f"\n{lang_name} ({lang_code}):")
    
    target_train_male = TRAIN_PER_LANG // 2
    target_train_female = TRAIN_PER_LANG - target_train_male
    target_test_male = TEST_PER_LANG // 2
    target_test_female = TEST_PER_LANG - target_test_male
    
    # Process Train - Male
    current = get_existing_counts(lang_code, "male", train_ai_dir)
    print(f"  Train Male: {current}/{target_train_male}")
    while current < target_train_male:
        text = random.choice(texts)
        if generate_sample(lang_code, "male", text, train_ai_dir, current):
            current += 1
            if current % 10 == 0: print(f"    Generated {current}/{target_train_male} male train")

    # Process Train - Female
    current = get_existing_counts(lang_code, "female", train_ai_dir)
    print(f"  Train Female: {current}/{target_train_female}")
    while current < target_train_female:
        text = random.choice(texts)
        if generate_sample(lang_code, "female", text, train_ai_dir, current):
            current += 1
            if current % 10 == 0: print(f"    Generated {current}/{target_train_female} female train")

    # Process Test - Male
    current = get_existing_counts(lang_code, "male", test_ai_dir)
    print(f"  Test Male: {current}/{target_test_male}")
    while current < target_test_male:
        text = random.choice(texts)
        if generate_sample(lang_code, "male", text, test_ai_dir, current):
            current += 1

    # Process Test - Female
    current = get_existing_counts(lang_code, "female", test_ai_dir)
    print(f"  Test Female: {current}/{target_test_female}")
    while current < target_test_female:
        text = random.choice(texts)
        if generate_sample(lang_code, "female", text, test_ai_dir, current):
            current += 1
            
    # Cleanup extras
    for gender, limit in [("male", target_train_male), ("female", target_train_female)]:
        files = sorted([f for f in os.listdir(train_ai_dir) if f.startswith(f"{lang_code}_{gender}_")])
        if len(files) > limit:
            print(f"  Cleaning up {len(files) - limit} extra {gender} train files...")
            for f in files[limit:]:
                os.remove(os.path.join(train_ai_dir, f))
                
    for gender, limit in [("male", target_test_male), ("female", target_test_female)]:
        files = sorted([f for f in os.listdir(test_ai_dir) if f.startswith(f"{lang_code}_{gender}_")])
        if len(files) > limit:
            print(f"  Cleaning up {len(files) - limit} extra {gender} test files...")
            for f in files[limit:]:
                os.remove(os.path.join(test_ai_dir, f))

# Cleanup temp folder
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)

print("\n" + "=" * 60)
print("AI Voice Generation Complete!")
print("=" * 60)
