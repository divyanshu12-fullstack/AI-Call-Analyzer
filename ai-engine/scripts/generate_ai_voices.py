import asyncio
import csv
import os
import random
import re
from datetime import datetime

# Avoid torchcodec/libtorchcodec dependency issues on Windows by preferring
# the legacy torchaudio path. Must be set before torch/torchaudio import.
os.environ.setdefault("TORCHAUDIO_USE_PYTORCH_CODEC", "0")
os.environ.setdefault("TORCHAUDIO_USE_TORCHCODEC", "0")

import torchaudio
from functools import partial

# Prefer non-torchcodec backend to avoid missing DLLs on Windows.
for backend in ("soundfile", "sox_io"):
    try:
        torchaudio.set_audio_backend(backend)
        break
    except Exception:
        continue

_orig_torchaudio_load = torchaudio.load
torchaudio.load = partial(_orig_torchaudio_load, backend="soundfile")

import torch
from TTS.api import TTS

try:
    import edge_tts
    _EDGE_TTS_AVAILABLE = True
except Exception:
    edge_tts = None
    _EDGE_TTS_AVAILABLE = False

# Define languages and their ISO codes
LANGUAGES = {
    # "english": "en",  # Already generated - skipped
    # "hindi": "hi",    # Already generated - skipped
    "tamil": "ta",
    "malayalam": "ml",
    "telugu": "te",
}

# Sample texts for each language (expand for variety)
TEXTS = {
    # English and Hindi texts already generated - skipped
    # "english": [
    #     "This is a verification call regarding your account.",
    # ],
    "english": [
        "This is a verification call regarding your account.",
        "Your account will be suspended unless you confirm your details.",
        "Please verify your identity to avoid service interruption.",
        "You have an outstanding balance that must be cleared today.",
        "We detected unusual activity on your account.",
        "Your card has been blocked for security reasons.",
        "Confirm your one-time password to proceed.",
        "Your bank account needs immediate verification.",
        "A refund is pending; confirm your details now.",
        "Your package is held due to an address issue.",
        "There is a legal notice in your name; respond urgently.",
        "Your phone line will be deactivated within 24 hours.",
        "We need your details to complete the verification.",
        "Press one to speak with the verification team.",
        "Do not share this code with anyone.",
        "We are calling from customer support regarding your account.",
        "Please confirm your UPI or bank details.",
        "Your KYC is incomplete and must be updated today.",
        "A suspicious transaction was detected; confirm immediately.",
        "Your account will be locked if you do not respond.",
        "This call is from the fraud prevention department.",
        "Your account shows a chargeback request; verify now.",
        "We need to verify your device to continue service.",
        "Your account will be verified in the next five minutes.",
        "Please update your billing address to avoid penalties.",
        "Your email was compromised; confirm the reset code.",
        "Your insurance claim requires verification of identity.",
        "Your wallet is flagged; confirm your details.",
        "We will close your account today if not verified.",
        "Your payment failed; update your details now.",
        "You are eligible for a refund; confirm your bank.",
        "Your SIM will be blocked; confirm your identity.",
        "Press two to verify your account now.",
        "Your account is under review; immediate response required.",
        "We detected an international transaction attempt.",
        "Confirm your account number for verification.",
        "Your profile needs re-verification.",
        "This is a final reminder to update your KYC.",
        "Please verify your PAN and bank details.",
        "You must act now to avoid account closure.",
    ],
    "tamil": [
        "உங்கள் கணக்கில் சந்தேகமான நடவடிக்கை கண்டறியப்பட்டது.",
        "உங்கள் கணக்கு தற்காலிகமாக முடக்கப்படும்; தயவுசெய்து உறுதிப்படுத்தவும்.",
        "உங்கள் அடையாளத்தைச் சரிபார்க்க வேண்டும்.",
        "உங்கள் KYC இன்று புதுப்பிக்கப்பட வேண்டும்.",
        "உங்கள் கார்டு பாதுகாப்புக்காக தடை செய்யப்பட்டுள்ளது.",
        "உங்கள் OTP ஐ உறுதிப்படுத்தவும்.",
        "உங்கள் வங்கி கணக்கு சரிபார்ப்பு தேவை.",
        "திருப்பிச் செலுத்தல் நிலுவையில் உள்ளது; விவரங்களை உறுதிப்படுத்தவும்.",
        "உங்கள் பார்சல் முகவரி பிரச்சனை காரணமாக தடுக்கப்பட்டுள்ளது.",
        "உங்கள் தொலைபேசி சேவை 24 மணி நேரத்தில் நிறுத்தப்படும்.",
        "சந்தேகமான பரிவர்த்தனை கண்டறியப்பட்டுள்ளது; உடனடி உறுதிப்படுத்தல் தேவை.",
        "உங்கள் கணக்கு பூட்டப்படுவதைத் தவிர்க்கவும்.",
        "உங்கள் விவரங்களை சரிபார்க்க வேண்டும்.",
        "உறுதிப்படுத்துதல் குழுவுடன் பேச ஒன்றை அழுத்தவும்.",
        "இந்த குறியீட்டை யாருடனும் பகிர வேண்டாம்.",
        "உங்கள் கணக்கைப் பற்றி பேச அழைத்தோம்.",
        "உங்கள் UPI அல்லது வங்கி விவரங்களை உறுதிப்படுத்தவும்.",
        "உங்கள் சேவை இடைநிறுத்தம் தவிர்க்க உடனே பதிலளிக்கவும்.",
        "உங்கள் கணக்கில் நிலுவை இருப்பு உள்ளது; இன்று செலுத்தவும்.",
        "உங்கள் பெயரில் சட்ட அறிவிப்பு உள்ளது; உடனே பதிலளிக்கவும்.",
        "மோசடி தடுப்பு துறையிலிருந்து அழைக்கப்படுகிறது.",
        "உங்கள் கணக்கு மதிப்பாய்வில் உள்ளது; உடனடி பதில் தேவை.",
        "உங்கள் கட்டணம் தோல்வி; விவரங்களை புதுப்பிக்கவும்.",
        "உங்கள் எண் சரிபார்க்கப்பட வேண்டும்.",
        "உங்கள் சிம் தடை செய்யப்படும்; அடையாளத்தை உறுதிப்படுத்தவும்.",
        "உங்கள் கணக்கு இன்று மூடப்படும்; உடனே உறுதிப்படுத்தவும்.",
        "உங்கள் கணக்கில் சர்வதேச பரிவர்த்தனை முயற்சி கண்டறியப்பட்டது.",
        "உங்கள் வங்கி விவரங்களை மீண்டும் சரிபார்க்க வேண்டும்.",
        "உங்கள் பான் மற்றும் வங்கி விவரங்களை உறுதிப்படுத்தவும்.",
        "உங்கள் கடன் கார்டு பாதுகாப்புக்காக நிறுத்தப்பட்டுள்ளது.",
        "உங்கள் காப்பீட்டு கோரிக்கை சரிபார்ப்பு தேவை.",
        "உங்கள் பதிவு செய்யப்பட்ட முகவரியைப் புதுப்பிக்கவும்.",
        "உங்கள் மின்னஞ்சல் பாதுகாப்புக்காக மீட்டமைப்பு தேவை.",
        "உங்கள் கணக்கு மீண்டும் சரிபார்க்கப்பட வேண்டும்.",
        "அவசர உறுதிப்படுத்தல் இல்லையெனில் சேவை நிறுத்தப்படும்.",
        "உங்கள் கணக்கு எண் உறுதிப்படுத்தல் தேவை.",
        "உங்கள் UPI பின் பகிர வேண்டாம்.",
        "உங்கள் பாக்கெட்/வாலெட் பாதுகாப்பு சரிபார்ப்பு தேவை.",
        "உங்கள் கணக்கு உறுதிப்படுத்தல் இன்று முடிக்க வேண்டும்.",
        "கணக்கு முடக்கம் தவிர்க்க இப்போது பதிலளிக்கவும்.",
    ],
    "hindi": [
        "आपके खाते में संदिग्ध गतिविधि पाई गई है।",
        "आपका खाता अस्थायी रूप से निलंबित किया जाएगा; कृपया पुष्टि करें।",
        "आपकी पहचान सत्यापित करना आवश्यक है।",
        "आपका KYC आज अपडेट होना चाहिए।",
        "आपका कार्ड सुरक्षा कारणों से ब्लॉक किया गया है।",
        "कृपया अपना OTP सत्यापित करें।",
        "आपके बैंक खाते का सत्यापन आवश्यक है।",
        "रिफंड लंबित है; कृपया विवरण पुष्टि करें।",
        "पता समस्या के कारण आपकी पार्सल रोक दी गई है।",
        "आपकी फोन सेवा 24 घंटे में बंद हो जाएगी।",
        "संदिग्ध लेन-देन मिला है; तुरंत पुष्टि करें।",
        "यदि आप जवाब नहीं देंगे तो आपका खाता लॉक हो जाएगा।",
        "हमें आपके विवरण की पुष्टि करनी है।",
        "सत्यापन टीम से बात करने के लिए एक दबाएँ।",
        "इस कोड को किसी के साथ साझा न करें।",
        "हम आपके खाते के संबंध में ग्राहक सहायता से बात कर रहे हैं।",
        "कृपया अपना UPI या बैंक विवरण पुष्टि करें।",
        "सेवा बाधित होने से बचाने के लिए तुरंत जवाब दें।",
        "आपके खाते में बकाया राशि है; आज भुगतान करें।",
        "आपके नाम पर कानूनी नोटिस है; तुरंत प्रतिक्रिया दें।",
        "यह कॉल फ्रॉड प्रिवेंशन विभाग से है।",
        "आपके खाते की समीक्षा चल रही है; तत्काल पुष्टि करें।",
        "आपका भुगतान असफल हुआ; विवरण अपडेट करें।",
        "आपका नंबर सत्यापित करना आवश्यक है।",
        "आपकी सिम ब्लॉक हो जाएगी; पहचान सत्यापित करें।",
        "आपका खाता आज बंद किया जाएगा; तुरंत पुष्टि करें।",
        "आपके खाते से अंतरराष्ट्रीय लेन-देन का प्रयास हुआ है।",
        "कृपया अपने बैंक विवरण दोबारा सत्यापित करें।",
        "कृपया अपना पैन और बैंक विवरण पुष्टि करें।",
        "आपका क्रेडिट कार्ड सुरक्षा कारणों से रोक दिया गया है।",
        "आपके बीमा दावे के लिए पहचान सत्यापन आवश्यक है।",
        "कृपया अपना पंजीकृत पता अपडेट करें।",
        "आपका ईमेल हैक हुआ; रीसेट कोड पुष्टि करें।",
        "आपके खाते का पुनः सत्यापन आवश्यक है।",
        "तत्काल सत्यापन न होने पर सेवा बंद होगी।",
        "कृपया अपना खाता नंबर सत्यापित करें।",
        "अपना UPI पिन किसी से साझा न करें।",
        "आपके वॉलेट की सुरक्षा जांच आवश्यक है।",
        "आपका KYC सत्यापन आज पूरा करना होगा।",
        "खाता लॉक होने से बचाने के लिए तुरंत प्रतिक्रिया दें।",
    ],
    "malayalam": [
        "നിങ്ങളുടെ അക്കൗണ്ടിൽ സംശയാസ്പദ പ്രവർത്തനം കണ്ടെത്തി.",
        "നിങ്ങളുടെ അക്കൗണ്ട് താൽക്കാലികമായി സസ്പെൻഡ് ചെയ്യപ്പെടും; സ്ഥിരീകരിക്കുക.",
        "നിങ്ങളുടെ തിരിച്ചറിയൽ സ്ഥിരീകരണം ആവശ്യമാണ്.",
        "നിങ്ങളുടെ KYC ഇന്ന് പുതുക്കണം.",
        "സുരക്ഷാ കാരണങ്ങളാൽ നിങ്ങളുടെ കാർഡ് ബ്ലോക്ക് ചെയ്തിട്ടുണ്ട്.",
        "ദയവായി നിങ്ങളുടെ OTP സ്ഥിരീകരിക്കുക.",
        "നിങ്ങളുടെ ബാങ്ക് അക്കൗണ്ട് സ്ഥിരീകരണം ആവശ്യമാണ്.",
        "റിഫണ്ട് നിലനില്ക്കുന്നു; വിശദാംശങ്ങൾ സ്ഥിരീകരിക്കുക.",
        "വിലാസ പ്രശ്നം മൂലം നിങ്ങളുടെ പാർസൽ തടഞ്ഞിട്ടുണ്ട്.",
        "നിങ്ങളുടെ ഫോൺ സേവനം 24 മണിക്കൂറിനകം നിർത്തും.",
        "സംശയാസ്പദ ഇടപാട് കണ്ടെത്തി; ഉടൻ സ്ഥിരീകരിക്കുക.",
        "നിങ്ങൾ പ്രതികരിക്കാത്ത പക്ഷം നിങ്ങളുടെ അക്കൗണ്ട് ലോക്ക് ചെയ്യും.",
        "നിങ്ങളുടെ വിവരങ്ങൾ സ്ഥിരീകരിക്കണം.",
        "സ്ഥിരീകരണ ടീമുമായി സംസാരിക്കാൻ ഒന്ന് അമർത്തുക.",
        "ഈ കോഡ് ആരോടും പങ്കുവെയ്ക്കരുത്.",
        "നിങ്ങളുടെ അക്കൗണ്ടിനെ കുറിച്ച് കസ്റ്റമർ സപ്പോർട്ടിൽ നിന്ന് വിളിക്കുകയാണ്.",
        "ദയവായി നിങ്ങളുടെ UPI അല്ലെങ്കിൽ ബാങ്ക് വിവരങ്ങൾ സ്ഥിരീകരിക്കുക.",
        "സേവനം തടസ്സപ്പെടുന്നത് ഒഴിവാക്കാൻ ഉടൻ പ്രതികരിക്കുക.",
        "നിങ്ങളുടെ അക്കൗണ്ടിൽ കുടിശ്ശികയുണ്ട്; ഇന്ന് പേയ്‌മെന്റ് ചെയ്യുക.",
        "നിങ്ങളുടെ പേരിൽ നിയമ നോട്ടീസ് ഉണ്ട്; ഉടൻ പ്രതികരിക്കുക.",
        "ഇത് ഫ്രോഡ് പ്രിവെൻഷൻ വിഭാഗത്തിൽ നിന്നുള്ള കോളാണ്.",
        "നിങ്ങളുടെ അക്കൗണ്ട് റിവ്യൂവിലാണ്; ഉടൻ സ്ഥിരീകരിക്കുക.",
        "നിങ്ങളുടെ പേയ്‌മെന്റ് പരാജയപ്പെട്ടു; വിശദാംശങ്ങൾ പുതുക്കുക.",
        "നിങ്ങളുടെ നമ്പർ സ്ഥിരീകരിക്കണം.",
        "നിങ്ങളുടെ സിം ബ്ലോക്ക് ചെയ്യും; തിരിച്ചറിയൽ സ്ഥിരീകരിക്കുക.",
        "നിങ്ങളുടെ അക്കൗണ്ട് ഇന്ന് അടയ്ക്കും; ഉടൻ സ്ഥിരീകരിക്കുക.",
        "നിങ്ങളുടെ അക്കൗണ്ടിൽ അന്തർദേശീയ ഇടപാട് ശ്രമം കണ്ടെത്തി.",
        "നിങ്ങളുടെ ബാങ്ക് വിവരങ്ങൾ വീണ്ടും സ്ഥിരീകരിക്കുക.",
        "ദയവായി നിങ്ങളുടെ പാൻ, ബാങ്ക് വിവരങ്ങൾ സ്ഥിരീകരിക്കുക.",
        "സുരക്ഷാ കാരണങ്ങളാൽ നിങ്ങളുടെ ക്രെഡിറ്റ് കാർഡ് തടഞ്ഞിട്ടുണ്ട്.",
        "നിങ്ങളുടെ ഇൻഷുറൻസ് ക്ലെയിംക്ക് തിരിച്ചറിയൽ സ്ഥിരീകരണം ആവശ്യമാണ്.",
        "ദയവായി രജിസ്റ്റർ ചെയ്ത വിലാസം പുതുക്കുക.",
        "നിങ്ങളുടെ ഇമെയിൽ ഹാക്ക് ചെയ്തു; റീസെറ്റ് കോഡ് സ്ഥിരീകരിക്കുക.",
        "നിങ്ങളുടെ അക്കൗണ്ട് വീണ്ടും സ്ഥിരീകരിക്കണം.",
        "ഉടൻ സ്ഥിരീകരിക്കില്ലെങ്കിൽ സേവനം നിർത്തും.",
        "ദയവായി നിങ്ങളുടെ അക്കൗണ്ട് നമ്പർ സ്ഥിരീകരിക്കുക.",
        "നിങ്ങളുടെ UPI പിൻ ആരോടും പങ്കുവെയ്ക്കരുത്.",
        "നിങ്ങളുടെ വാലറ്റ് സുരക്ഷാ പരിശോധന ആവശ്യമാണ്.",
        "നിങ്ങളുടെ KYC ഇന്ന് പൂർത്തിയാക്കണം.",
        "അക്കൗണ്ട് ലോക്ക് ഒഴിവാക്കാൻ ഉടൻ പ്രതികരിക്കുക.",
    ],
    "telugu": [
        "మీ ఖాతాలో అనుమానాస్పద కార్యకలాపం గుర్తించబడింది.",
        "మీ ఖాతాను తాత్కాలికంగా నిలిపివేస్తారు; దయచేసి నిర్ధారించండి.",
        "మీ గుర్తింపును ధృవీకరించాలి.",
        "మీ KYC ఈ రోజే అప్డేట్ చేయాలి.",
        "భద్రత కారణాల వల్ల మీ కార్డ్ బ్లాక్ చేయబడింది.",
        "దయచేసి మీ OTP ని నిర్ధారించండి.",
        "మీ బ్యాంక్ ఖాతా ధృవీకరణ అవసరం.",
        "రిఫండ్ పెండింగ్‌లో ఉంది; వివరాలు నిర్ధారించండి.",
        "చిరునామా సమస్య కారణంగా మీ పార్సెల్ నిలిపివేశారు.",
        "మీ ఫోన్ సేవ 24 గంటల్లో నిలిపివేయబడుతుంది.",
        "అనుమానాస్పద లావాదేవీ గుర్తించబడింది; వెంటనే నిర్ధారించండి.",
        "మీరు స్పందించకపోతే మీ ఖాతా లాక్ అవుతుంది.",
        "మీ వివరాలను ధృవీకరించాలి.",
        "ధృవీకరణ బృందంతో మాట్లాడడానికి 1 నొక్కండి.",
        "ఈ కోడ్‌ను ఎవరికీ చెప్పవద్దు.",
        "మీ ఖాతా గురించి కస్టమర్ సపోర్ట్ నుండి కాల్ చేస్తున్నాము.",
        "దయచేసి మీ UPI లేదా బ్యాంక్ వివరాలను నిర్ధారించండి.",
        "సేవ అంతరాయం తప్పించడానికి వెంటనే స్పందించండి.",
        "మీ ఖాతాలో బకాయి ఉంది; ఈ రోజే చెల్లించండి.",
        "మీ పేరుతో చట్టపరమైన నోటీస్ ఉంది; వెంటనే స్పందించండి.",
        "ఇది ఫ్రాడ్ ప్రివెన్షన్ విభాగం నుండి కాల్.",
        "మీ ఖాతా సమీక్షలో ఉంది; వెంటనే నిర్ధారించండి.",
        "మీ చెల్లింపు విఫలమైంది; వివరాలను అప్డేట్ చేయండి.",
        "మీ నంబర్‌ను ధృవీకరించాలి.",
        "మీ సిమ్ బ్లాక్ అవుతుంది; గుర్తింపును ధൃవీకరించండి.",
        "మీ ఖాతా ఈ రోజే మూసివేయబడుతుంది; వెంటనే నిర్ధారించండి.",
        "మీ ఖాతాలో అంతర్జాతీయ లావాదేవీ ప్రయత్నం జరిగింది.",
        "దయచేసి మీ బ్యాంక్ వివరాలను మళ్లీ ధృవీకరించండి.",
        "దయచేసి మీ పాన్ మరియు బ్యాంక్ వివరాలను నిర్ధారించండి.",
        "భద్రత కారణాల వల్ల మీ క్రెడిట్ కార్డు నిలిపివేశారు.",
        "మీ బీమా క్లెయిమ్‌కు గుర్తింపు ధృవీకరణ అవసరం.",
        "దయచేసి మీ నమోదు చేసిన చిరునామాను అప్డేట్ చేయండి.",
        "మీ ఇమెయిల్ హ్యాక్ అయింది; రీసెట్ కోడ్ నిర్ధారించండి.",
        "మీ ఖాతాను మళ్లీ ధృవీకరించాలి.",
        "వెంటనే నిర్ధారించకపోతే సేవ నిలిపివేస్తారు.",
        "దయచేసి మీ ఖాతా నంబర్‌ను ధൃవీకరించండి.",
        "మీ UPI పిన్ ఎవరికీ చెప్పవద్దు.",
        "మీ వాలెట్ భద్రతా తనిఖీ అవసరం.",
        "మీ KYC ఈ రోజే పూర్తి చేయాలి.",
        "ఖాతా లాక్ కాకుండా వెంటనే స్పందించండి.",
    ],
}

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
DEVICE = "cpu"  # default; overwritten in _select_device()

TRAIN_COUNT = 200
TEST_COUNT = 50
SEED = 42

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train", "ai")
TEST_DIR = os.path.join(BASE_DIR, "data", "test", "ai")
METADATA_PATH = os.path.join(BASE_DIR, "data", "ai_generated_metadata.csv")

SPEAKER_WAV_DIR = os.getenv("SPEAKER_WAV_DIR")
DEFAULT_SPEAKER_DIRS = [
    os.path.join(BASE_DIR, "data", "train", "human"),
    os.path.join(BASE_DIR, "data", "test", "human"),
]
SPEED_MIN = 0.92
SPEED_MAX = 1.08

XTTS_SUPPORTED_LANGS = {
    "en",
    "hi",
    "es",
    "fr",
    "de",
    "it",
    "pt",
    "pl",
    "tr",
    "ru",
    "nl",
    "cs",
    "ar",
    "zh-cn",
    "hu",
    "ko",
    "ja",
}

EDGE_TTS_VOICES = {
    "ta": "ta-IN-PallaviNeural",
    "ml": "ml-IN-SobhanaNeural",
    "te": "te-IN-ShrutiNeural",
    "hi": "hi-IN-SwaraNeural",
    "en": "en-US-AriaNeural",
}

EDGE_TTS_OUTPUT_FORMAT = "audio-16khz-32kbitrate-mono-mp3"


def _sanitize_text(text: str) -> str:
    """Remove digits to avoid num2words failures in XTTS multilingual cleaners."""
    cleaned = re.sub(r"\d+", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or text


async def _edge_tts_save(text: str, filepath: str, lang_code: str) -> None:
    """Generate TTS using Edge TTS (Microsoft cloud-based)."""
    if not _EDGE_TTS_AVAILABLE:
        raise RuntimeError(
            "edge-tts is not installed. Install with 'pip install edge-tts'."
        )
    voice = EDGE_TTS_VOICES.get(lang_code, EDGE_TTS_VOICES.get("en"))
    try:
        communicate = edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(filepath)
    except Exception as exc:
        print(f"    Edge TTS failed: {exc}")
        raise


async def _generate_split_async(
    tts: TTS,
    split: str,
    count: int,
    speaker_wavs: list[str],
) -> None:
    """Generate audio using Edge TTS for Tamil, Malayalam, Telugu."""
    metadata_rows: list[dict] = []
    for lang_name, lang_code in LANGUAGES.items():
        out_dir = TRAIN_DIR if split == "train" else TEST_DIR
        lang_dir = os.path.join(out_dir, lang_name)
        print(f"Generating {count} samples for {lang_name} ({split})...")

        for idx in range(count):
            raw_text = random.choice(TEXTS[lang_name])
            text = _sanitize_text(raw_text)
            speed = round(random.uniform(SPEED_MIN, SPEED_MAX), 3)
            
            filename = f"{lang_name}_ai_{split}_{idx + 1:04d}.wav"
            filepath = os.path.join(lang_dir, filename)

            try:
                await _edge_tts_save(text, filepath, lang_code)
            except Exception as exc:
                print(f"    Error generating {filename}: {exc}")
                continue

            metadata_rows.append({
                "timestamp": datetime.utcnow().isoformat(),
                "split": split,
                "language": lang_name,
                "text": raw_text,
                "path": filepath,
                "speaker_wav": "",
                "speed": speed,
                "model": "edge-tts",
                "device": "cloud",
            })

            if (idx + 1) % 50 == 0:
                print(f"  Generated {idx + 1}/{count} for {lang_name} ({split})")

    _write_metadata(metadata_rows)


def _select_device() -> str:
    """Pick a CUDA device only if an NVIDIA GPU is present; otherwise CPU."""
    if not torch.cuda.is_available():
        return "cpu"

    gpu_count = torch.cuda.device_count()
    for idx in range(gpu_count):
        props = torch.cuda.get_device_properties(idx)
        name = props.name.lower()
        if "nvidia" in name or "geforce" in name or "gtx" in name:
            torch.cuda.set_device(idx)
            return f"cuda:{idx}"

    # No NVIDIA-like device found; avoid unexpected AMD/other backends.
    return "cpu"


def _list_speaker_wavs(speaker_dirs: list[str]) -> list[str]:
    wavs: list[str] = []
    for speaker_dir in speaker_dirs:
        if not speaker_dir or not os.path.isdir(speaker_dir):
            continue
        for root, _, files in os.walk(speaker_dir):
            for name in files:
                if name.lower().endswith(".wav"):
                    wavs.append(os.path.join(root, name))
    return wavs


def _ensure_dirs() -> None:
    for base in (TRAIN_DIR, TEST_DIR):
        os.makedirs(base, exist_ok=True)
        for lang_name in LANGUAGES:
            os.makedirs(os.path.join(base, lang_name), exist_ok=True)


def _write_metadata(rows: list[dict]) -> None:
    if not rows:
        return
    is_new = not os.path.exists(METADATA_PATH)
    with open(METADATA_PATH, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "timestamp",
                "split",
                "language",
                "text",
                "path",
                "speaker_wav",
                "speed",
                "model",
                "device",
            ],
        )
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def _generate_split(
    tts: TTS,
    split: str,
    count: int,
    speaker_wavs: list[str],
) -> None:
    metadata_rows: list[dict] = []
    for lang_name, lang_code in LANGUAGES.items():
        out_dir = TRAIN_DIR if split == "train" else TEST_DIR
        lang_dir = os.path.join(out_dir, lang_name)
        print(f"Generating {count} samples for {lang_name} ({split})...")

        for idx in range(count):
            raw_text = random.choice(TEXTS[lang_name])
            text = _sanitize_text(raw_text)
            speed = round(random.uniform(SPEED_MIN, SPEED_MAX), 3)
            speaker_wav = random.choice(speaker_wavs) if speaker_wavs else None

            filename = f"{lang_name}_ai_{split}_{idx + 1:04d}.wav"
            filepath = os.path.join(lang_dir, filename)

            use_xtts = lang_code in XTTS_SUPPORTED_LANGS
            if use_xtts:
                try:
                    tts.tts_to_file(
                        text=text,
                        file_path=filepath,
                        language=lang_code,
                        speaker_wav=speaker_wav,
                        speed=speed,
                        split_sentences=True,
                    )
                except Exception as exc:
                    print(
                        f"  XTTS failed for {lang_name} at {idx + 1}: {exc}. "
                        "Falling back to Edge TTS."
                    )
                    use_xtts = False

            if not use_xtts:
                asyncio.run(_edge_tts_save(text, filepath, lang_code))

            metadata_rows.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "split": split,
                    "language": lang_name,
                    "text": raw_text,
                    "path": filepath,
                    "speaker_wav": speaker_wav or "",
                    "speed": speed,
                    "model": MODEL_NAME,
                    "device": DEVICE,
                }
            )

            if (idx + 1) % 25 == 0:
                print(f"  Generated {idx + 1}/{count} for {lang_name} ({split})")

    _write_metadata(metadata_rows)


def _summarize_counts() -> None:
    for split, base_dir in (("train", TRAIN_DIR), ("test", TEST_DIR)):
        print(f"\nSummary ({split}):")
        for lang_name in LANGUAGES:
            lang_dir = os.path.join(base_dir, lang_name)
            if not os.path.isdir(lang_dir):
                print(f"  {lang_name}: 0")
                continue
            count = sum(
                1
                for name in os.listdir(lang_dir)
                if name.lower().endswith(".wav")
            )
            print(f"  {lang_name}: {count}")


def main() -> None:
    global DEVICE
    random.seed(SEED)
    _ensure_dirs()

    speaker_dirs = [SPEAKER_WAV_DIR] if SPEAKER_WAV_DIR else []
    if not speaker_dirs:
        speaker_dirs = DEFAULT_SPEAKER_DIRS
    speaker_wavs = _list_speaker_wavs(speaker_dirs)
    if speaker_wavs:
        print(f"Found {len(speaker_wavs)} speaker WAVs for variance.")
    else:
        raise RuntimeError(
            "No speaker WAVs found. Provide SPEAKER_WAV_DIR or place WAVs in "
            "data/train/human or data/test/human. XTTS requires speaker_wav."
        )

    device = _select_device()
    DEVICE = device
    if device.startswith("cuda"):
        idx = int(device.split(":")[1])
        props = torch.cuda.get_device_properties(idx)
        print(f"Using device: {device} ({props.name})")
    else:
        print(f"Using device: {device}")

    if not _EDGE_TTS_AVAILABLE:
        print("Edge TTS not installed; fallback will fail if XTTS errors.")

    tts = TTS(MODEL_NAME).to(device)

    # Generate only Tamil, Malayalam, Telugu (English and Hindi already generated)
    print("\nGenerating only Tamil, Malayalam, Telugu (English/Hindi skipped)...\n")
    asyncio.run(_generate_split_async(tts, "train", TRAIN_COUNT, speaker_wavs))
    asyncio.run(_generate_split_async(tts, "test", TEST_COUNT, speaker_wavs))
    _summarize_counts()
    print("\nAI voice generation completed!")


if __name__ == "__main__":
    main()
