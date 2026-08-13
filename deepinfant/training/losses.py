"""
Loss Functions for Imbalanced Multiclass Cry Classification.
Supports CrossEntropyLoss, Class-Weighted CrossEntropy, and Focal Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing severe class imbalance in audio classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: [batch, num_classes], targets: [batch]
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def get_loss_function(
    loss_type: str = "weighted_cross_entropy",
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0
) -> nn.Module:
    """Factory function for training loss functions."""
    loss_type = loss_type.lower()
    
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
    elif loss_type == "weighted_cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == "focal_loss":
        return FocalLoss(alpha=class_weights, gamma=focal_gamma)
    else:
        raise ValueError(f"Unknown loss type: '{loss_type}'")
