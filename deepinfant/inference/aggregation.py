"""
Prediction Aggregation and Temporal Smoothing Module.
Combines multiple sliding window predictions and applies exponential temporal smoothing.
Formula: P_smooth(t) = alpha * P(t) + (1 - alpha) * P_smooth(t-1)
"""

import numpy as np
from typing import List, Dict, Any


class TemporalPredictionSmoother:
    """
    Applies exponential moving average (EMA) smoothing to continuous real-time probability vectors.
    Prevents erratic visual jumpiness in real-time UI.
    """
    
    def __init__(self, alpha: float = 0.35, num_classes: int = 9):
        self.alpha = alpha
        self.num_classes = num_classes
        self.smoothed_probs: np.ndarray = np.ones(num_classes) / num_classes

    def update(self, new_probs: np.ndarray) -> np.ndarray:
        """Updates internal state with new probability vector."""
        if len(new_probs) != self.num_classes:
            raise ValueError(f"Expected probability array of length {self.num_classes}, got {len(new_probs)}")
            
        self.smoothed_probs = self.alpha * new_probs + (1.0 - self.alpha) * self.smoothed_probs
        # Normalize sum to 1.0
        self.smoothed_probs = self.smoothed_probs / np.sum(self.smoothed_probs)
        return self.smoothed_probs

    def reset(self):
        """Resets smoothed vector to uniform distribution."""
        self.smoothed_probs = np.ones(self.num_classes) / self.num_classes


def aggregate_window_predictions(
    window_probs_list: List[np.ndarray],
    weights: Optional[List[float]] = None
) -> np.ndarray:
    """
    Aggregates probability vectors across multiple sliding audio windows.
    P_final(class) = average(P_window_1, P_window_2, ..., P_window_n)
    """
    if not window_probs_list:
        raise ValueError("Cannot aggregate empty list of window probabilities.")
        
    probs_matrix = np.array(window_probs_list) # [num_windows, num_classes]
    
    if weights is not None:
        weights = np.array(weights) / np.sum(weights)
        aggregated = np.sum(probs_matrix * weights[:, np.newaxis], axis=0)
    else:
        aggregated = np.mean(probs_matrix, axis=0)
        
    # Renormalize to ensure sum == 1.0
    return aggregated / np.sum(aggregated)
