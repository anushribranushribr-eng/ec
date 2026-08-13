#!/usr/bin/env python3
"""
CLI Script: Dataset Preparation & Subject-Level Stratification.
Scans raw dataset directory, parses metadata filenames, creates subject-level train/val/test splits,
and saves metadata CSV files.
"""

import os
import argparse
import pandas as pd
from deepinfant.datasets.metadata import parse_cry_filename
from deepinfant.datasets.splitting import subject_level_split, verify_split_integrity


def main():
    parser = argparse.ArgumentParser(description="Prepare DeepInfant V2 Dataset Splits")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Directory containing audio files")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory for metadata CSVs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Scanning directory: {args.data_dir}...")
    records = []
    
    if os.path.exists(args.data_dir):
        for root, _, files in os.walk(args.data_dir):
            for file in files:
                if file.endswith((".wav", ".caf", ".3gp", ".mp3", ".m4a")):
                    full_path = os.path.join(root, file)
                    parsed = parse_cry_filename(file)
                    parsed["file_path"] = full_path
                    records.append(parsed)
                    
    if not records:
        print("No audio files found. Generating synthetic dataset metadata for demonstration...")
        # Generate representative dataset metadata for testing
        labels = ["belly_pain", "burping", "cold_hot", "discomfort", "hungry", "lonely", "scared", "tired", "unknown"]
        for i in range(120):
            baby_id = f"BABY_{i // 4 + 1:03d}"
            lbl = labels[i % len(labels)]
            records.append({
                "filename": f"sample_{i+1:03d}_{lbl}.wav",
                "file_path": f"data/raw/sample_{i+1:03d}_{lbl}.wav",
                "label": lbl,
                "baby_id": baby_id,
                "gender": "male" if i % 2 == 0 else "female",
                "age_months": (i % 6) + 1
            })
            
    df = pd.DataFrame(records)
    print(f"Total audio samples indexed: {len(df)}")
    print(f"Total unique subjects (baby_id): {df['baby_id'].nunique()}")
    
    print("\nClass distribution:")
    print(df["label"].value_counts())
    
    # Subject-level split
    train_df, val_df, test_df = subject_level_split(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=args.seed)
    
    integrity = verify_split_integrity(train_df, val_df, test_df)
    print("\nSplit Integrity Check:")
    print(f"  Passed Zero-Subject Leakage Check: {integrity['integrity_passed']}")
    print(f"  Train: {len(train_df)} samples ({integrity['train_subjects']} subjects)")
    print(f"  Val:   {len(val_df)} samples ({integrity['val_subjects']} subjects)")
    print(f"  Test:  {len(test_df)} samples ({integrity['test_subjects']} subjects)")
    
    # Save CSVs
    train_df.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(args.output_dir, "test.csv"), index=False)
    df.to_csv(os.path.join(args.output_dir, "full_dataset.csv"), index=False)
    print(f"\nMetadata CSVs saved to '{args.output_dir}/'")


if __name__ == "__main__":
    main()
