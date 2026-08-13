#!/usr/bin/env python3
"""
CLI Script: Evaluate DeepInfant V2 Checkpoint.
Evaluates model on held-out test dataset split and generates per-class metrics and confusion matrix.
"""

import os
import argparse
import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader

from deepinfant.datasets.dataset import InfantCryDataset, CLASSES
from deepinfant.models.factory import build_model
from deepinfant.training.metrics import calculate_metrics, CLASSES


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepInfant V2 Checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory containing processed test.csv")
    args = parser.parse_args()

    test_csv = os.path.join(args.data_dir, "test.csv")
    if not os.path.exists(test_csv):
        print("test.csv not found. Running synthetic evaluation demonstration...")
        labels = CLASSES
        records = [{"filename": f"test_{i}.wav", "label": labels[i % 9], "baby_id": f"TEST_BABY_{i//2}"} for i in range(27)]
        test_df = pd.DataFrame(records)
    else:
        test_df = pd.read_csv(test_csv)

    config = {
        "audio": {"sample_rate": 16000, "n_mels": 128, "target_duration_sec": 4.0},
        "model": {"backbone": "cnn_lstm", "num_classes": 9}
    }

    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        config = checkpoint.get("config", config)
        model = build_model(config)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded checkpoint from: {args.checkpoint}")
    else:
        print(f"Checkpoint not found at '{args.checkpoint}'. Instantiating fresh model for evaluation...")
        model = build_model(config)

    model.eval()
    test_dataset = InfantCryDataset(test_df, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.numpy())
            y_pred.extend(preds.numpy())

    metrics = calculate_metrics(y_true, y_pred, CLASSES)

    print("\n" + "=" * 65)
    print("DEEPINFANT V2 EVALUATION REPORT (HELD-OUT TEST SPLIT)")
    print("=" * 65)
    print(f"Overall Accuracy:  {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1 Score:   {metrics['macro_f1']:.4f}")
    print(f"Weighted F1 Score:{metrics['weighted_f1']:.4f}")
    print("-" * 65)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
    print("-" * 65)
    for cls_name, vals in metrics["per_class"].items():
        print(f"{cls_name:<15} {vals['precision']:<12.4f} {vals['recall']:<12.4f} {vals['f1_score']:<12.4f} {vals['support']:<8}")
    print("=" * 65)


if __name__ == "__main__":
    main()
