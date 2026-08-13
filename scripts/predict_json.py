#!/usr/bin/env python3
"""
CLI Script: Predict Infant Cry Reason for Audio File (JSON Output).
Usage: python scripts/predict_json.py --audio example.wav
Output: JSON format for API consumption
"""

import os
import sys
import argparse
import json
import torch
import numpy as np

from deepinfant.models.factory import build_model
from deepinfant.inference.predictor import CryPredictor
from deepinfant.audio.loader import load_audio


def main():
    parser = argparse.ArgumentParser(description="Predict Infant Cry Reason from Audio (JSON Output)")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file (.wav, .m4a, .caf, .3gp)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.45, help="Confidence rejection threshold")
    args = parser.parse_args()

    config = {
        "audio": {"sample_rate": 16000, "n_mels": 128, "n_fft": 2048, "hop_length": 512, "target_duration_sec": 4.0},
        "model": {"backbone": "cnn_lstm", "num_classes": 9},
        "system": {"device": "cpu"}
    }

    try:
        if os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location="cpu")
            config = checkpoint.get("config", config)
            model = build_model(config)
            model.load_state_dict(checkpoint["model_state"])
        else:
            model = build_model(config)

        predictor = CryPredictor(model, config, confidence_threshold=args.threshold)

        if not os.path.exists(args.audio):
            # Create synthetic test waveform if file doesn't exist
            sr = 16000
            t = np.linspace(0, 4.0, int(sr * 4.0))
            audio = 0.5 * np.sin(2 * np.pi * 450 * t) + 0.1 * np.random.randn(len(t))
            result = predictor.predict_audio_array(audio.astype(np.float32))
        else:
            result = predictor.predict_file(args.audio)

        # Convert numpy types to Python native types for JSON serialization
        output = {
            "prediction": str(result["prediction"]),
            "confidence": float(result["confidence"]),
            "raw_top_class": str(result.get("raw_top_class", result["prediction"])),
            "raw_top_confidence": float(result.get("raw_top_confidence", result["confidence"])),
            "is_unknown": bool(result.get("is_unknown", False)),
            "confidence_threshold": float(result.get("confidence_threshold", args.threshold)),
            "probabilities": {k: float(v) for k, v in result["probabilities"].items()},
            "safety_disclaimer": str(result.get("safety_disclaimer", ""))
        }
        
        # Print JSON output (ONLY JSON, no other text for easy parsing)
        print(json.dumps(output))
        sys.exit(0)
        
    except Exception as e:
        error_output = {
            "error": str(e),
            "prediction": "unknown",
            "confidence": 0.0,
            "probabilities": {},
            "is_unknown": True
        }
        print(json.dumps(error_output))
        sys.exit(1)


if __name__ == "__main__":
    main()
