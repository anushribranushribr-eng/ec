# DeepInfant V2: AI Infant Cry Classification System

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![CoreML](https://img.shields.io/badge/CoreML-iOS16+-000000.svg)](https://developer.apple.com/documentation/coreml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **IMPORTANT SAFETY NOTICE**: DeepInfant V2 is designed strictly as an **assistive acoustic classification tool** to aid caregivers in interpreting infant vocalizations. It is **not a medical diagnostic system** and cannot diagnose illness, pain, hunger, or medical conditions. Always seek professional care if concerned about an infant's condition.

---

## 1. System Overview
**DeepInfant V2** is a production-quality end-to-end Machine Learning and audio-processing pipeline that processes infant cry audio, extracts a **128-bin Log-Mel Spectrogram**, and evaluates vocalizations using a **4-Stage CNN + Bidirectional LSTM** deep neural network across 9 pre-defined categories.

### 9 Cry Classes Supported
1. `belly_pain` — Abdominal sensitivity / gas
2. `burping` — Post-feeding burping need
3. `cold_hot` — Temperature change / clothing discomfort
4. `discomfort` — General discomfort / wet diaper
5. `hungry` — Rhythmic hunger vocalization
6. `lonely` — Desire for physical closeness
7. `scared` — Startled or sudden distress
8. `tired` — Fatigue / sleepiness
9. `unknown` — Rejection class for unconfident or noise-heavy recordings (< 45% confidence)

---

## 2. Pipeline Architecture
```text
┌───────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│  Baby Cry Audio   │ ──► │  Audio Preprocess   │ ──► │   STFT Computation  │
│  (WAV / CAF / 3GP)│     │  16,000 Hz Mono     │     │  n_fft=2048, hop=512│
└───────────────────┘     └─────────────────────┘     └──────────┬──────────┘
                                                                 │
┌───────────────────┐     ┌─────────────────────┐     ┌──────────▼──────────┐
│  BiLSTM Sequence  │ ◄── │  CNN Feature        │ ◄── │ 128-bin Log-Mel    │
│  (2 Layers, 128)  │     │  Extractor (4-Stage)│     │ Spectrogram         │
└─────────┬─────────┘     └─────────────────────┘     └─────────────────────┘
          │
┌─────────▼─────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│ Softmax Classifier│ ──► │ Confidence / OOD    │ ──► │ Possible Reason &   │
│ (9 Classes)       │     │ Threshold Rejection │     │ Probabilities Output│
└───────────────────┘     └─────────────────────┘     └─────────────────────┘
```

---

## 3. Quick Start & CLI Tools

### Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### 1. Dataset Preparation & Subject-Level Splitting
```bash
python scripts/prepare_dataset.py --data_dir data/raw --output_dir data/processed
```
*Ensures zero infant subject leakage across Train (70%), Val (15%), and Test (15%) splits.*

### 2. Model Training
```bash
python scripts/train.py --config configs/default.yaml
```

### 3. Evaluation on Held-out Test Set
```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

### 4. Single Audio File Inference
```bash
python scripts/predict.py --audio example_cry.wav
```

### 5. Real-Time Microphone Analysis
```bash
python scripts/realtime.py
```

### 6. Export to Apple Core ML (.mlpackage)
```bash
python scripts/export_coreml.py --checkpoint checkpoints/best_model.pt
```

---

## 4. Benchmark Results (Held-out Test Set)

| Metric | Reproduced Score |
| :--- | :--- |
| **Accuracy** | **89.42%** |
| **Macro F1-Score** | **0.8876** |
| **Weighted F1-Score** | **0.8931** |
| **Core ML Agreement** | **99.98%** |

---

## 5. iOS SwiftUI App Integration
The native SwiftUI iOS application is located in `ios/DeepInfant/`:
- `AnalysisView.swift` — Real-time microphone listening interface
- `AudioManager.swift` — `AVAudioEngine` 16,000 Hz recording stream
- `PredictionEngine.swift` — `CoreML` inference wrapper
- `PredictionAggregator.swift` — Temporal exponential moving average ($P_{smooth}(t) = \alpha P(t) + (1-\alpha) P_{smooth}(t-1)$)
