import os
import random
from TTS.api import TTS

# Define languages and their ISO codes
languages = {
    "english": "en",
    "tamil": "ta",
    "hindi": "hi",
    "malayalam": "ml",
    "telugu": "te"
}

# Sample texts for each language (expand for variety)
texts = {
    "english": [
        "Hello, how are you today?",
        "This is a test of the AI voice generation system.",
        "I am generating synthetic speech for machine learning.",
        "The weather is nice outside.",
        "Can you hear me clearly?"
    ],
    "tamil": [
        "வணக்கம், நீங்கள் எப்படி இருக்கிறீர்கள்?",
        "இது AI குரல் உருவாக்க சிஸ்டத்தின் சோதனை.",
        "நான் இயந்திர கற்றலுக்காக செயற்கை பேச்சை உருவாக்குகிறேன்.",
        "வெளியே வானிலை நன்றாக இருக்கிறது.",
        "நீங்கள் என்னை தெளிவாக கேட்க முடிகிறதா?"
    ],
    "hindi": [
        "नमस्ते, आप कैसे हैं?",
        "यह AI आवाज निर्माण प्रणाली का परीक्षण है।",
        "मैं मशीन लर्निंग के लिए सिंथेटिक स्पीच उत्पन्न कर रहा हूं।",
        "बाहर मौसम अच्छा है।",
        "क्या आप मुझे स्पष्ट रूप से सुन सकते हैं?"
    ],
    "malayalam": [
        "നമസ്കാരം, നിങ്ങൾ എങ്ങനെ ആണ്?",
        "ഇത് AI വോയ്സ് ജനറേഷൻ സിസ്റ്റത്തിന്റെ ടെസ്റ്റ് ആണ്.",
        "ഞാൻ മെഷീൻ ലേർണിംഗിനായി സിന്തറ്റിക് സ്പീച്ച് ജനറേറ്റ് ചെയ്യുന്നു.",
        "പുറത്ത് കാലാവസ്ഥ നല്ലതാണ്.",
        "നിങ്ങൾക്ക് എന്നെ വ്യക്തമായി കേൾക്കാൻ കഴിയുമോ?"
    ],
    "telugu": [
        "నమస్కారం, మీరు ఎలా ఉన్నారు?",
        "ఇది AI వాయిస్ జనరేషన్ సిస్టమ్ యొక్క టెస్ట్.",
        "నేను మెషిన్ లెర్నింగ్ కోసం సింథటిక్ స్పీచ్ జనరేట్ చేస్తున్నాను.",
        "వెలుపల వాతావరణం మంచిది.",
        "మీరు నన్ను స్పష్టంగా వినగలరా?"
    ]
}

# Initialize TTS model (XTTS v2)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda" if torch.cuda.is_available() else "cpu")

# Output directory
output_dir = "data/train/ai"
os.makedirs(output_dir, exist_ok=True)

# Generate 250 samples per language
samples_per_lang = 250

for lang_name, lang_code in languages.items():
    lang_dir = os.path.join(output_dir, lang_name)
    os.makedirs(lang_dir, exist_ok=True)
    
    print(f"Generating {samples_per_lang} samples for {lang_name}...")
    
    for i in range(samples_per_lang):
        # Randomly select a text
        text = random.choice(texts[lang_name])
        
        # Generate filename
        filename = f"{lang_name}_ai_{i+1:03d}.wav"
        filepath = os.path.join(lang_dir, filename)
        
        # Generate speech (without reference for now, using default speaker)
        tts.tts_to_file(text=text, file_path=filepath, language=lang_code)
        
        if (i+1) % 50 == 0:
            print(f"  Generated {i+1}/{samples_per_lang} for {lang_name}")

print("AI voice generation completed!")