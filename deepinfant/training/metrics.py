"""
Evaluation Metrics Module for DeepInfant V2.
Calculates Accuracy, Precision, Recall, F1-Score, Macro F1, Weighted F1, and Confusion Matrix.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, Any, List

CLASSES = [
    "belly_pain",
    "burping",
    "cold_hot",
    "discomfort",
    "hungry",
    "lonely",
    "scared",
    "tired",
    "unknown"
]


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASSES
) -> Dict[str, Any]:
    """
    Computes comprehensive evaluation metrics for multiclass classification.
    
    Returns dictionary containing:
      - accuracy
      - macro_f1
      - weighted_f1
      - per_class (dict of precision, recall, f1 for each class)
      - confusion_matrix (2D numpy list)
    """
    acc = float(accuracy_score(y_true, y_pred))
    
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    # Per-class metrics
    p_class, r_class, f1_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(class_names))), zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    
    per_class_dict = {}
    for i, name in enumerate(class_names):
        per_class_dict[name] = {
            "precision": float(p_class[i]),
            "recall": float(r_class[i]),
            "f1_score": float(f1_class[i]),
            "support": int(support[i])
        }
        
    return {
        "accuracy": acc,
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "per_class": per_class_dict,
        "confusion_matrix": cm.tolist()
    }
