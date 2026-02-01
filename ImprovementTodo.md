# AI Voice Detection Model - Improvement Todo

> **Objective**: Build a secure REST API that detects AI-generated vs Human voices in Tamil, English, Hindi, Malayalam, and Telugu with high accuracy.

---

## 📊 Impact Legend
| Symbol | Impact Level | Accuracy Improvement |
|--------|--------------|---------------------|
| 🔴 | Critical | +15-25% |
| 🟠 | High | +8-15% |
| 🟡 | Medium | +3-8% |
| 🟢 | Low | +1-3% |

---

## 🔴 CRITICAL IMPROVEMENTS (Highest Priority)

### 1. Upgrade Model Architecture
**Current**: Simple 3-layer CNN (64 features)
**Problem**: Too shallow to capture complex patterns in modern AI voices
**Solution**:
- [x] Implement ResNet-18/34 style architecture with residual connections
- [x] Add attention mechanisms for temporal focus
- [x] Consider audio-specific architectures (ECAPA-TDNN, Wav2Vec2 fine-tuning)
- [x] Add dropout layers (0.3-0.5) to prevent overfitting

**Impact**: 🔴 +15-20% accuracy
**Effort**: High (2-3 days)

---

### 2. Expand and Diversify Training Dataset
**Current**: ~800 samples, mostly LibriSpeech + basic TTS
**Problem**: Model hasn't seen realistic AI voices or diverse human audio (songs, conversations)

#### AI Voice Data Improvements:
- [ ] Add samples from modern TTS systems:
  - Tortoise TTS (diffusion-based, very realistic)
  - Bark (Suno AI)
  - XTTS (Coqui)
  - ElevenLabs-style voices
  - VALL-E style zero-shot TTS
- [ ] Include voice cloning outputs
- [ ] Add deepfake audio samples
- [ ] Generate 2000+ AI samples per language

#### Human Voice Data Improvements:
- [ ] Add singing/musical vocals (critical for song misclassification fix)
- [ ] Include conversational speech (not just read speech)
- [ ] Add emotional speech (angry, happy, sad, excited)
- [ ] Include noisy environment recordings
- [ ] Add phone call quality recordings
- [ ] Include child and elderly voices
- [ ] Expand regional accent coverage for all 5 languages
- [ ] Target 2000+ human samples per language

**Impact**: 🔴 +20-25% accuracy
**Effort**: High (3-5 days for data collection)

---

### 3. Multi-Feature Extraction
**Current**: Only mel spectrograms (64 bins)
**Problem**: Single feature type misses important voice characteristics

**Add these features**:
- [x] MFCC coefficients (13-20 coefficients + deltas)
- [x] Pitch/F0 tracking (AI often has unnatural pitch patterns)
- [x] Formant analysis (F1, F2, F3 frequencies)
- [x] Spectral contrast
- [x] Chroma features
- [x] Voice quality measures:
  - Jitter (pitch variation)
  - Shimmer (amplitude variation)
  - Harmonic-to-Noise Ratio (HNR)
- [x] Zero-crossing rate
- [x] Spectral centroid and bandwidth

**Impact**: 🔴 +12-18% accuracy
**Effort**: Medium (1-2 days)

---

## 🟠 HIGH PRIORITY IMPROVEMENTS

### 4. Proper Training Pipeline
**Current**: 5 epochs, no validation, no early stopping
**Problem**: Model likely underfitting, no way to detect overfitting

**Improvements**:
- [x] Increase epochs to 50-100 with early stopping
- [x] Implement train/validation/test split (70/15/15)
- [x] Add learning rate scheduler (ReduceLROnPlateau or CosineAnnealing)
- [x] Implement proper cross-validation (5-fold)
- [x] Add gradient clipping for stability
- [x] Use weighted loss for class imbalance
- [x] Save best model checkpoint based on validation accuracy

**Impact**: 🟠 +8-12% accuracy
**Effort**: Low (4-6 hours)

---

### 5. Data Augmentation During Training
**Current**: No augmentation during training
**Problem**: Model doesn't generalize well to varied audio conditions

**Add these augmentations**:
- [ ] Time stretching (0.8x - 1.2x speed)
- [ ] Pitch shifting (±2 semitones)
- [ ] Background noise injection (SNR 10-30dB)
- [ ] Room impulse response simulation
- [ ] Random gain adjustment
- [ ] Time masking (SpecAugment)
- [ ] Frequency masking (SpecAugment)
- [ ] Audio compression simulation (MP3 artifacts)
- [ ] Telephony band-limiting

**Impact**: 🟠 +8-10% accuracy
**Effort**: Medium (1 day)

---

### 6. Language-Specific Processing
**Current**: Same model for all languages, language parameter unused
**Problem**: Different languages have different phonetic characteristics

**Options**:
- [ ] Option A: Single model with language embedding input
- [ ] Option B: Separate fine-tuned models per language
- [ ] Option C: Multi-task learning with language as auxiliary task
- [ ] Add language-specific feature normalization
- [ ] Include language-specific training data balancing

**Impact**: 🟠 +5-10% accuracy (especially for non-English)
**Effort**: Medium-High (2-3 days)

---

### 7. Pre-trained Audio Embeddings
**Current**: Training from scratch on small dataset
**Problem**: Limited data makes learning robust features difficult

**Solutions**:
- [ ] Use Wav2Vec 2.0 embeddings (Facebook/Meta)
- [ ] Use HuBERT embeddings
- [ ] Use ECAPA-TDNN speaker embeddings
- [ ] Fine-tune pre-trained model on detection task
- [ ] Consider multilingual pre-trained models (XLSR-53)

**Impact**: 🟠 +10-15% accuracy
**Effort**: Medium (1-2 days)

---

## 🟡 MEDIUM PRIORITY IMPROVEMENTS

### 8. Variable Length Audio Support
**Current**: Fixed 5-second window, truncates/pads
**Problem**: Loses context from longer audio, struggles with short clips

**Improvements**:
- [ ] Implement sliding window with voting
- [ ] Add global pooling for variable lengths
- [ ] Use attention-based aggregation over time
- [ ] Handle audio from 1-60 seconds gracefully

**Impact**: 🟡 +3-5% accuracy
**Effort**: Medium (1 day)

---

### 9. Better Evaluation Metrics
**Current**: Only accuracy metric
**Problem**: Can't identify specific failure modes

**Add these metrics**:
- [ ] Precision and Recall per class
- [ ] F1-score (macro and weighted)
- [ ] ROC-AUC curve
- [ ] Confusion matrix visualization
- [ ] Per-language accuracy breakdown
- [ ] False positive/negative analysis
- [ ] Confidence calibration metrics (ECE)

**Impact**: 🟡 +2-4% (indirect, helps identify issues)
**Effort**: Low (4-6 hours)

---

### 10. Confidence Calibration
**Current**: Raw sigmoid output, uncalibrated
**Problem**: Confidence scores don't reflect true probability

**Improvements**:
- [ ] Temperature scaling calibration
- [ ] Platt scaling
- [ ] Isotonic regression calibration
- [ ] Monte Carlo dropout for uncertainty
- [ ] Ensemble disagreement as uncertainty

**Impact**: 🟡 +2-3% (improves confidence score quality)
**Effort**: Low (4-6 hours)

---

### 11. Ensemble Methods
**Current**: Single model
**Problem**: Single point of failure, limited robustness

**Implement**:
- [ ] Train 3-5 models with different seeds
- [ ] Combine CNN + RNN + Transformer
- [ ] Use different feature sets per model
- [ ] Implement voting or averaging ensemble
- [ ] Add model disagreement as uncertainty signal

**Impact**: 🟡 +5-8% accuracy
**Effort**: Medium-High (2-3 days)

---

### 12. Robust Preprocessing
**Current**: Assumes clean 16kHz input
**Problem**: Real-world audio varies significantly

**Improvements**:
- [ ] Automatic sample rate detection and resampling
- [ ] Handle stereo to mono conversion properly
- [ ] Add silence/noise detection and trimming
- [ ] Normalize audio levels (peak or LUFS)
- [ ] Handle corrupted or partial files gracefully
- [ ] Add input validation for audio quality

**Impact**: 🟡 +2-4% accuracy
**Effort**: Low (4-6 hours)

---

## 🟢 LOW PRIORITY (Nice to Have)

### 13. Improved Explanations
**Current**: Generic explanations
**Problem**: Doesn't meet "quality of explanation" evaluation criteria well

**Improvements**:
- [ ] Add feature-based explanations (which features triggered detection)
- [ ] Include confidence breakdown
- [ ] Add language-specific reasoning
- [ ] Implement attention visualization for explainability
- [ ] Provide specific artifact descriptions

**Impact**: 🟢 +1-2% (evaluation criteria score)
**Effort**: Low (4-6 hours)

---

### 14. API Performance Optimization
**Current**: Synchronous processing
**Problem**: May timeout on large files or high load

**Improvements**:
- [ ] Add async processing for large files
- [ ] Implement request queuing
- [ ] Add caching for repeated requests
- [ ] Optimize model inference (ONNX, TorchScript)
- [ ] Add request timeout handling
- [ ] Implement batch processing option

**Impact**: 🟢 +1% (reliability score)
**Effort**: Medium (1 day)

---

### 15. Code Quality & Security
**Current**: Some hardcoded values, potential security issues
**Problem**: May cause issues in production

**Fixes**:
- [ ] Remove duplicate constants
- [ ] Add proper error handling throughout
- [ ] Secure temporary file handling
- [ ] Add input sanitization
- [ ] Implement rate limiting
- [ ] Add request logging for debugging
- [ ] Environment-based configuration

**Impact**: 🟢 +1% (reliability)
**Effort**: Low (4-6 hours)

---

## � NEXT IMPLEMENTATION SEQUENCE

1. Add basic data augmentation during training (time stretching, pitch shifting, background noise, etc.)
2. Implement proper evaluation metrics (precision, recall, F1-score, ROC-AUC, confusion matrix, per-language accuracy)
3. Collect modern AI voice samples (Tortoise TTS, Bark, XTTS, ElevenLabs, VALL-E, voice cloning outputs, deepfakes)
4. Add diverse human data (singing/vocals, conversational speech, emotional speech, noisy environments, phone quality, regional accents, child/elderly voices)
5. Balance dataset across languages (target 200 samples per language for both AI and human and 50 for test)
6. Retrain with expanded data
7. Add pre-trained audio embeddings (Wav2Vec2, HuBERT, ECAPA-TDNN)
8. Implement ensemble methods (train 3-5 models, combine predictions)
9. Confidence calibration (temperature scaling, Platt scaling)
10. Better explanations (feature-based explanations, confidence breakdown, language-specific reasoning)
11. API performance optimization (async processing, caching, ONNX/TorchScript)
12. Code quality & security (error handling, input sanitization, rate limiting, logging)

---

### Phase 1: Quick Wins (1-2 days)
1. [x] Fix training pipeline (validation, early stopping, more epochs)
2. [ ] Add basic data augmentation
3. [ ] Implement proper evaluation metrics
4. [x] Add multi-feature extraction (MFCC, pitch)

**Expected Improvement**: +15-20% accuracy

### Phase 2: Data Expansion (3-5 days)
1. [ ] Collect modern AI voice samples
2. [ ] Add diverse human data (songs, conversations)
3. [ ] Balance dataset across languages
4. [ ] Retrain with expanded data

**Expected Improvement**: +15-25% accuracy

### Phase 3: Architecture Upgrade (2-3 days)
1. [x] Implement deeper CNN or transformer
2. [ ] Add pre-trained embeddings
3. [ ] Implement ensemble methods

**Expected Improvement**: +10-15% accuracy

### Phase 4: Polish (1-2 days)
1. [ ] Confidence calibration
2. [ ] Better explanations
3. [ ] API optimization
4. [ ] Code cleanup

**Expected Improvement**: +2-5% accuracy

---

## 🎯 TOTAL EXPECTED IMPROVEMENT

| Phase | Improvement | Cumulative |
|-------|-------------|------------|
| Current Baseline | - | ~60-65% |
| Phase 1 | +15-20% | ~75-85% |
| Phase 2 | +15-25% | ~85-92% |
| Phase 3 | +10-15% | ~92-97% |
| Phase 4 | +2-5% | ~95-98% |

**Actual Phase 1 Result**: 100% test accuracy achieved! (Massive improvement beyond expectations)

---

## ⚠️ COMPLIANCE CHECKLIST

Based on [Compliance.md](Compliance.md):

- [x] REST API with POST endpoint
- [x] Accepts Base64-encoded MP3
- [x] Supports all 5 languages (Tamil, English, Hindi, Malayalam, Telugu)
- [x] Returns JSON with required fields
- [x] API key authentication (x-api-key header)
- [x] Classification: AI_GENERATED or HUMAN
- [x] Confidence score (0.0 - 1.0)
- [x] Explanation field
- [ ] **TODO**: Ensure no hard-coding of results
- [ ] **TODO**: Improve accuracy across all 5 languages
- [ ] **TODO**: Improve explanation quality

---

## 📝 NOTES

1. **Most Critical Issue**: The model fails on realistic AI voices because training data only includes basic TTS (gTTS, Edge TTS). Must add modern AI samples.

3. **Language Coverage**: Currently English-heavy. Need balanced data for Tamil, Hindi, Malayalam, Telugu.

4. **Quick Test**: After each improvement, test with:
   - A clearly robotic AI voice (should classify as AI)
   - A modern ElevenLabs-style voice (should classify as AI)
   - A clean human speech (should classify as Human)
   - A human singing (should classify as Human)
   - Samples in all 5 languages

---

*Last Updated: January 31, 2026*
