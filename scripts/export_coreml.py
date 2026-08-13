#!/usr/bin/env python3
"""
CLI Script: Export DeepInfant V2 Model to Apple Core ML (.mlpackage).
Usage: python scripts/export_coreml.py --checkpoint checkpoints/best_model.pt
"""

import os
import argparse
import torch

from deepinfant.models.factory import build_model
from deepinfant.export.coreml import export_pytorch_to_coreml, validate_coreml_against_pytorch


def main():
    parser = argparse.ArgumentParser(description="Export DeepInfant V2 to Apple Core ML")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to PyTorch checkpoint")
    parser.add_argument("--output", type=str, default="DeepInfant_V2.mlpackage", help="Output Core ML package path")
    args = parser.parse_args()

    config = {
        "audio": {"sample_rate": 16000, "n_mels": 128, "target_duration_sec": 4.0},
        "model": {"backbone": "cnn_lstm", "num_classes": 9}
    }

    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        config = checkpoint.get("config", config)
        model = build_model(config)
        model.load_state_dict(checkpoint["model_state"])
    else:
        model = build_model(config)

    print(f"Exporting PyTorch DeepInfant model to Core ML package: '{args.output}'...")
    output_path = export_pytorch_to_coreml(model, output_path=args.output)
    print(f"Successfully exported Core ML model package to: {output_path}")

    print("\nRunning Core ML vs PyTorch Numerical Consistency Validation...")
    val_report = validate_coreml_against_pytorch(model, output_path, test_samples=20)
    
    print("-" * 50)
    print(f"Status: {val_report['status']}")
    print(f"Test Samples: {val_report['num_test_samples']}")
    print(f"Max Absolute Probability Error: {val_report['max_absolute_error']:.6f}")
    print(f"Class Agreement Rate: {val_report['class_agreement_percentage']:.2f}%")
    print("-" * 50)


if __name__ == "__main__":
    main()
