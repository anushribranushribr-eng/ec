#!/usr/bin/env python3
"""
CLI Script: Real-Time Microphone Infant Cry Analysis.
Usage: python scripts/realtime.py
"""

import time
import torch
import numpy as np

from deepinfant.models.factory import build_model
from deepinfant.inference.predictor import CryPredictor
from deepinfant.inference.realtime import RealtimeInferencePipeline


def main():
    print("Starting DeepInfant V2 Real-Time Microphone Audio Analyzer...")
    
    config = {
        "audio": {"sample_rate": 16000, "n_mels": 128, "target_duration_sec": 4.0},
        "model": {"backbone": "cnn_lstm", "num_classes": 9},
        "system": {"device": "cpu"},
        "classes": ["belly_pain", "burping", "cold_hot", "discomfort", "hungry", "lonely", "scared", "tired", "unknown"]
    }
    
    model = build_model(config)
    predictor = CryPredictor(model, config, confidence_threshold=0.45)
    pipeline = RealtimeInferencePipeline(predictor, buffer_duration_sec=4.0, sr=16000)

    print("Pipeline active. Simulating microphone audio stream buffer...")
    print("Press Ctrl+C to stop.")
    
    try:
        # Simulate audio streaming loop
        for step in range(10):
            chunk = np.random.randn(4000).astype(np.float32) * 0.05
            res = pipeline.push_audio_chunk(chunk)
            print(f"[Step {step+1:02d}] Cry Detected: {res['cry_detected']} | Top Prediction: {res['prediction']} | Confidence: {res['confidence']*100:.1f}%")
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\nReal-time inference stopped.")


if __name__ == "__main__":
    main()
