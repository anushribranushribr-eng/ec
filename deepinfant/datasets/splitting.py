"""
Dataset Splitting Module (Subject-Level Stratification).
Strictly prevents data leakage by ensuring all audio clips from a given infant (baby_id)
belong exclusively to either train, validation, or test set.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List


def subject_level_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits DataFrame into Train, Val, Test sets based on subject IDs (baby_id).
    
    CRITICAL: Ensures NO baby_id occurs in more than one split!
    
    Args:
        df: DataFrame containing at least 'baby_id' and 'label' columns
        train_ratio: Fraction for training set (default: 0.70)
        val_ratio: Fraction for validation set (default: 0.15)
        test_ratio: Fraction for test set (default: 0.15)
        seed: Random seed for reproducibility
        
    Returns:
        train_df, val_df, test_df
    """
    if "baby_id" not in df.columns:
        raise ValueError("DataFrame must contain 'baby_id' column for subject-level splitting.")
        
    np.random.seed(seed)
    unique_subjects = df["baby_id"].unique()
    np.random.shuffle(unique_subjects)
    
    num_subjects = len(unique_subjects)
    num_train = int(np.round(train_ratio * num_subjects))
    num_val = int(np.round(val_ratio * num_subjects))
    
    train_subjects = set(unique_subjects[:num_train])
    val_subjects = set(unique_subjects[num_train : num_train + num_val])
    test_subjects = set(unique_subjects[num_train + num_val :])
    
    # Verify zero overlap
    assert len(train_subjects.intersection(val_subjects)) == 0
    assert len(train_subjects.intersection(test_subjects)) == 0
    assert len(val_subjects.intersection(test_subjects)) == 0
    
    train_df = df[df["baby_id"].isin(train_subjects)].copy().reset_index(drop=True)
    val_df = df[df["baby_id"].isin(val_subjects)].copy().reset_index(drop=True)
    test_df = df[df["baby_id"].isin(test_subjects)].copy().reset_index(drop=True)
    
    return train_df, val_df, test_df


def verify_split_integrity(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Dict[str, bool]:
    """Verifies that no subject ID is shared across splits."""
    train_babies = set(train_df["baby_id"])
    val_babies = set(val_df["baby_id"])
    test_babies = set(test_df["baby_id"])
    
    leak_train_val = len(train_babies.intersection(val_babies)) > 0
    leak_train_test = len(train_babies.intersection(test_babies)) > 0
    leak_val_test = len(val_babies.intersection(test_babies)) > 0
    
    has_leakage = leak_train_val or leak_train_test or leak_val_test
    
    return {
        "integrity_passed": not has_leakage,
        "leak_train_val": leak_train_val,
        "leak_train_test": leak_train_test,
        "leak_val_test": leak_val_test,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "train_subjects": len(train_babies),
        "val_subjects": len(val_babies),
        "test_subjects": len(test_babies)
    }
