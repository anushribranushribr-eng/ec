#!/usr/bin/env python3
"""
CLI Script: Predict Infant Cry Reason for Audio File.
Usage: python scripts/predict.py --audio example.wav
"""

import os
import argparse
import torch
import numpy as np

from deepinfant.models.factory import build_model
from deepinfant.inference.predictor import CryPredictor
from deepinfant.audio.loader import load_audio


def main():
    parser = argparse.ArgumentParser(description="Predict Infant Cry Reason from Audio")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file (.wav, .m4a, .caf, .3gp)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.45, help="Confidence rejection threshold")
    args = parser.parse_args()

    config = {
        "audio": {"sample_rate": 16000, "n_mels": 128, "target_duration_sec": 4.0},
        "model": {"backbone": "cnn_lstm", "num_classes": 9},
        "system": {"device": "cpu"}
    }

    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        config = checkpoint.get("config", config)
        model = build_model(config)
        model.load_state_dict(checkpoint["model_state"])
    else:
        model = build_model(config)

    predictor = CryPredictor(model, config, confidence_threshold=args.threshold)

    if not os.path.exists(args.audio):
        print(f"Audio file '{args.audio}' not found. Generating 4-second synthetic audio clip for prediction test...")
        # Create synthetic test waveform
        sr = 16000
        t = np.linspace(0, 4.0, int(sr * 4.0))
        audio = 0.5 * np.sin(2 * np.pi * 450 * t) + 0.1 * np.random.randn(len(t))
        result = predictor.predict_audio_array(audio.astype(np.float32))
    else:
        result = predictor.predict_file(args.audio)

    print("DeepInfant V2")
    print("-" * 30)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence'] * 100:.1f}%")
    print("\nProbabilities:")
    
    # Sort probabilities descending
    sorted_probs = sorted(result["probabilities"].items(), key=lambda x: x[1], reverse=True)
    for cls_name, prob in sorted_probs:
        print(f"{cls_name:<12} {prob * 100:>5.1f}%")
        
    print("\nNote:")
    print("This is an AI-generated acoustic prediction and is not a medical diagnosis.")


if __name__ == "__main__":
    main()
