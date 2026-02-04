# 🎙️ AI Voice Detector Engine (ResNet-18 Edition)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A professional-grade ResNet-Audio pipeline designed to detect synthetic speech across 5 languages with calibrated confidence.**

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [API Reference](#-api-reference) • [Model Details](#-model-details)

</div>

---

## 🌟 Features

- **Multilingual Support** — Specialized for English, Hindi, Tamil, Telugu, and Malayalam.
- **SE-ResNet Architecture** — **Squeeze-and-Excitation** Attention mechanism dynamically weights Pitch vs. Spectral features for context-aware detection.
- **Sliding Window Inference** — Analyzes entire long-form dialogues by scanning 5-second overlapping chunks.
- **Calibrated Confidence** — Implements Temperature Scaling to ensure confidence scores are statistically honest.
- **Real-World Robustness** — Trained with **Telephony Simulation** (Bandpass/Gain), **SpecAugment** (Masking), and **Channel Dropout** (Zeroing Pitch/ZCR) to prevent overfitting.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AI VOICE DETECTION PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────┐     ┌─────────────┐     ┌────────────────┐    ┌──────────┐   │
│   │ Audio    │────▶│ 11-Channel  │────▶│ Sliding Window │───▶│ Batch    │   │
│   │ (Base64) │     │ Feature Ext │     │ (5s Stride)    │    │ Inference│   │
│   └──────────┘     └─────────────┘     └────────────────┘    └──────────┘   │
│                                                                    │        │
│                                                                    ▼        │
│   ┌────────────┐     ┌────────────┐     ┌───────────┐     ┌─────────────┐   │
│   │ JSON       │◀────│ Temp Scale │◀────│ Max/Mean  │◀────│SE-ResNet-18 │   │
│   │ Response   │     │ Calibration│     │ Aggregator│     │ (Attention) │   │
│   └────────────┘     └────────────┘     └───────────┘     └─────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Anaconda / Miniconda (Recommended)

### Installation

```bash
cd ai-engine
pip install -r requirements.txt
```

### Full Pipeline Workflow

```bash
# 1. Generate Multilingual Dataset
python scripts/generate_daily_ai_voices.py

# 2. Train the ResNet Model
python train.py

# 3. Calibrate Confidence Scores
python calibrate.py

# 4. Evaluate Performances
python evaluate.py

# 5. Start the Production API
python api.py
```

---

## 📡 API Reference

### Detect Voice
```http
POST /api/voice-detection
Content-Type: application/json
x-api-key: YOUR_API_KEY

{
  "language": "Tamil",
  "audioFormat": "mp3",
  "audioBase64": "<base64_encoded_audio>"
}
```

**Response Format:**
```json
{
  "status": "success",
  "language": "Tamil",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.9654,
  "explanation": "Very high confidence (96.54%) - Clear synthetic speech patterns detected. High spectral uniformity...",
  "meta": {
    "windows_analyzed": 4,
    "max_ai_prob": 0.9821,
    "avg_ai_prob": 0.8542
  }
}
```

---

## 🧠 Model Details

### 11-Channel Feature Extraction
The model doesn't just look at a spectrogram. It extracts **11 acoustic channels** representing 113 unique features:
- **Mel Spectrogram** (64 bands)
- **MFCCs + Deltas** (26 channels)
- **F0 Pitch Tracking** (1 channel)
- **Spectral Contrast** (7 channels)
- **Chroma STFT** (12 channels)
- **ZCR, Centroid, Bandwidth** (3 channels)

### Architecture: SE-VoiceResNet
A custom deep residual network enhanced with **Squeeze-and-Excitation (SE) Blocks**:
- **Residual Blocks**: 4 Layers of basic blocks for deep feature learning.
- **Attention (SE)**: Adaptive average pooling blocks that learn to re-weight channels (e.g., ignoring Pitch if it's noisy) for every single inference.
- **Dropout**: 0.4 probability to prevent over-fitting.
- **Focal Loss**: Trained with Focal Loss to prioritize hard-to-classify examples over easy ones.

---

## 📊 Performance (Benchmark)

| Metric | Score |
|--------|-------|
| **Accuracy** | **97.08%** |
| **Recall (AI Detection)** | **100.00%** |
| **Precision (Human)** | **100.00%** |
| **ROC-AUC** | **0.9980** |

*Note: Benchmarked on a balanced set of 565 samples across 5 languages.*

---

## 📁 Project Structure

```
ai-engine/
├── 📄 api.py                  # Production FastAPI server
├── 📄 model.py                # VoiceResNet architecture
├── 📄 dataset.py              # Augmented Data Loader
├── 📄 train.py                # Training logic with early stopping
├── 📄 calibrate.py            # Calibration runner
├── 📄 calibration.py          # Temperature scaling implementation
├── 📄 inference.py            # Sliding window prediction engine
├── 📄 requirements.txt        # Dependencies
└── 📁 scripts/
    └── 📄 generate_daily_ai_voices.py  # Multilingual TTS generator
```

---

<div align="center">

**Built with ❤️ for robust synthetic voice detection.**

</div>
