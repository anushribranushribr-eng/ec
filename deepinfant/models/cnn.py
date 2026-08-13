"""
CNN Feature Extractor Backbone.
Extracts 2D acoustic features from Log-Mel Spectrogram input [batch, 1, 128, time_steps].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Standard 2D Convolutional Block with BatchNorm, ReLU, and MaxPool."""
    
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class CNNFeatureExtractor(nn.Module):
    """
    4-Stage CNN Feature Extractor Backbone.
    Input: [batch, 1, 128, time_steps]
    Output: Feature sequence [batch, time_steps_compressed, feature_dim]
    """
    
    def __init__(self, in_channels: int = 1, filters: list = None):
        super().__init__()
        if filters is None:
            filters = [32, 64, 128, 256]
            
        self.block1 = ConvBlock(in_channels, filters[0], pool=True) # 128 -> 64
        self.block2 = ConvBlock(filters[0], filters[1], pool=True)  # 64 -> 32
        self.block3 = ConvBlock(filters[1], filters[2], pool=True)  # 32 -> 16
        self.block4 = ConvBlock(filters[2], filters[3], pool=False) # 16 x time

        self.out_channels = filters[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, 1, 128, time_steps]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x) # shape: [batch, 256, 16, time_steps_reduced]
        
        # Collapse frequency dimension (16) into channel features
        batch, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous() # [batch, t, c, f]
        x = x.view(batch, t, c * f) # [batch, t, 256 * 16 = 4096]
        return x
