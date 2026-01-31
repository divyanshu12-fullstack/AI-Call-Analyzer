# 🎙️ AI Voice Detector Engine

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A CNN-based deep learning system that detects whether audio is AI-generated or human-spoken**

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [API Reference](#-api-reference) • [Model Details](#-model-details)

</div>

---

## 🌟 Features

- **Binary Classification** — Distinguishes between AI-generated and human speech
- **Multi-language Support** — Works with English, Hindi, Tamil, Telugu, Malayalam
- **Real-time Inference** — Fast predictions via REST API
- **Explainable Results** — Returns confidence scores with technical explanations
- **Lightweight Model** — CPU-friendly CNN architecture (~500KB)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         AI VOICE DETECTION PIPELINE                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────────┐  │
│   │  Audio   │────▶│  Decode  │────▶│  Resample│────▶│     Mel      │  │
│   │ (Base64) │     │  MP3/WAV │     │  16kHz   │     │ Spectrogram  │  │
│   └──────────┘     └──────────┘     └──────────┘     └──────────────┘  │
│                                                              │          │
│                                                              ▼          │
│   ┌──────────────┐     ┌──────────┐     ┌──────────────────────────┐   │
│   │    JSON      │◀────│ Threshold│◀────│      CNN Classifier      │   │
│   │   Response   │     │   >0.5   │     │   (3 Conv + FC layers)   │   │
│   └──────────────┘     └──────────┘     └──────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Response Format
```json
{
  "classification": "AI_GENERATED",
  "confidence": 0.9866,
  "explanation": "High spectral uniformity and lack of natural pitch variation strongly indicate synthetic speech generation."
}
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd AI-Call-Analyzer/ai-engine

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Training the Model

```bash
# 1. Download human voice samples (LibriSpeech)
python download_hf_human.py

# 2. Generate AI voice samples (using gTTS)
python generate_ai_voices.py

# 3. Train the CNN model
python train.py

# 4. Evaluate on test set
python evaluate.py
```

### Running the API Server

```bash
python api.py
```

The API will be available at `http://localhost:8000`

📖 **Interactive Docs**: http://localhost:8000/docs

---

## 📡 API Reference

### Health Check
```http
GET /
```

**Response:**
```json
{
  "status": "online",
  "service": "AI Voice Detector",
  "version": "1.0.0"
}
```

### Detect Voice
```http
POST /api/voice-detection
Content-Type: application/json
x-api-key: YOUR_API_KEY

{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "<base64_encoded_audio>"
}
```

**Response:**
```json
{
  "status": "success",
  "language": "English",
  "classification": "HUMAN" | "AI_GENERATED",
  "confidenceScore": 0.0 - 1.0,
  "explanation": "Technical reasoning for the classification"
}
```

**Error Response:**
```json
{
  "status": "error",
  "message": "Error description"
}
```

### Example Usage (Python)
```python
import base64
import requests

# Read and encode audio file
with open("sample.mp3", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

# Send request
response = requests.post(
    "http://localhost:8000/api/voice-detection",
    headers={"x-api-key": "your-api-key"},
    json={
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": audio_b64
    }
)

print(response.json())
```

### Example Usage (cURL)
```bash
curl -X POST http://localhost:8000/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAA..."
  }'
```

---

## 🧠 Model Details

### Architecture: VoiceCNN

| Layer | Type | Output Shape | Parameters |
|-------|------|-------------|------------|
| Input | - | (1, 64, 157) | - |
| Conv1 | Conv2d(1→16, 3×3) + ReLU + MaxPool | (16, 32, 78) | 160 |
| Conv2 | Conv2d(16→32, 3×3) + ReLU + MaxPool | (32, 16, 39) | 4,640 |
| Conv3 | Conv2d(32→64, 3×3) + ReLU + MaxPool | (64, 8, 19) | 18,496 |
| Pool | AdaptiveAvgPool2d(1,1) | (64, 1, 1) | - |
| FC | Linear(64→1) + Sigmoid | (1,) | 65 |

**Total Parameters:** ~23,361 (92KB)

### Feature Extraction

- **Input**: 16kHz mono audio (max 5 seconds)
- **Mel Spectrogram**: 64 mel bands
- **Normalization**: Zero-mean, unit-variance per sample
- **Fixed Length**: 157 frames (padded/trimmed)

### Training Configuration

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Binary Cross Entropy |
| Batch Size | 8 |
| Epochs | 20 |

---

## 📊 Performance

### Test Set Results

| Metric | Score |
|--------|-------|
| **Accuracy** | 92.5% |
| Human Precision | ~90% |
| AI Precision | ~95% |

### Confusion Matrix (on 40 test samples)

```
              Predicted
              Human    AI
Actual Human    18      2
       AI        1     19
```

> ⚠️ **Note**: Performance may vary on out-of-distribution audio (different languages, accents, TTS engines not seen during training).

---

## 📁 Project Structure

```
ai-engine/
├── 📄 api.py                  # FastAPI server with /detect endpoint
├── 📄 model.py                # VoiceCNN architecture definition
├── 📄 dataset.py              # PyTorch Dataset for audio loading
├── 📄 train.py                # Training script
├── 📄 evaluate.py             # Evaluation script
├── 📄 inference.py            # Inference module with VoiceDetector class
├── 📄 preprocess.py           # Audio preprocessing utilities
├── 📄 download_hf_human.py    # Script to download human voice samples
├── 📄 generate_ai_voices.py   # Script to generate AI voice samples
├── 📄 requirements.txt        # Python dependencies
├── 📄 voice_model.pth         # Trained model weights
└── 📁 data/
    ├── 📁 train/
    │   ├── 📁 human/          # Human voice training samples
    │   └── 📁 ai/             # AI voice training samples
    └── 📁 test/
        ├── 📁 human/          # Human voice test samples
        └── 📁 ai/             # AI voice test samples
```

---

## 🔬 Technical Details

### Why Mel Spectrograms?

Mel spectrograms provide a time-frequency representation that:
1. Mimics human auditory perception (mel scale)
2. Captures formant patterns distinctive to speech
3. Works well with 2D CNNs (image-like representation)
4. Reduces dimensionality compared to raw waveforms

### Detection Approach

The model learns to distinguish between:

| Human Speech | AI-Generated Speech |
|-------------|-------------------|
| Natural pitch variations | Consistent pitch patterns |
| Breathing artifacts | No breath sounds |
| Micro-pauses | Uniform pacing |
| Spectral irregularities | Smooth spectral contours |

### Explainability

Predictions include human-readable explanations based on:
- Confidence level (high/medium/low)
- Classification result (human/AI)
- Spectral characteristics observed

---

## 🛠️ Development

### Running Tests
```bash
python -m pytest tests/ -v
```

### Code Formatting
```bash
pip install black isort
black .
isort .
```

### Adding New TTS Engines for Training

Edit `generate_ai_voices.py` to include samples from:
- Azure TTS
- Amazon Polly
- ElevenLabs
- Coqui TTS
- Bark

More diverse AI samples improve generalization.

---

## 📋 Requirements

```txt
torch>=2.0.0
torchaudio>=2.0.0
librosa>=0.10.0
numpy>=1.24.0
soundfile>=0.12.0
fastapi>=0.100.0
uvicorn>=0.23.0
datasets>=2.14.0
huggingface-hub>=0.16.0
gTTS>=2.3.0
```

---

## 🚧 Limitations

1. **Training Data Bias**: Model trained primarily on English LibriSpeech + gTTS. May underperform on:
   - Other languages
   - Different TTS engines (ElevenLabs, Bark, etc.)
   - Phone-quality audio
   - Background noise

2. **Adversarial Robustness**: Not tested against adversarial attacks or heavily post-processed AI audio.

3. **Short Audio**: Clips under 1 second may have reduced accuracy.

---

## 🗺️ Roadmap

- [ ] Add multi-language training data
- [ ] Implement attention mechanisms
- [ ] Add confidence calibration
- [ ] Create browser-based demo
- [ ] Docker containerization
- [ ] Model quantization for edge deployment

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [LibriSpeech](https://www.openslr.org/12/) for human speech samples
- [gTTS](https://gtts.readthedocs.io/) for AI voice generation
- [librosa](https://librosa.org/) for audio processing
- [PyTorch](https://pytorch.org/) for deep learning framework

---

<div align="center">

**Built with ❤️ for detecting synthetic voices**

</div>
