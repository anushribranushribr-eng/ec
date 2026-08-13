"""
Model Factory Module.
Instantiates DeepInfant models based on configuration (cnn_lstm, resnet18_lstm, resnet34_lstm).
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from deepinfant.models.cnn_lstm import DeepInfantCNNLSTM


class ResNetAudioLSTM(nn.Module):
    """
    ResNet backbone + BiLSTM for Audio Classification.
    """
    def __init__(self, num_classes: int = 9, dropout: float = 0.3):
        super().__init__()
        # Custom lightweight 2D ResNet-style blocks for 128-bin Mel-spectrogram
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet Block 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        
        # ResNet Block 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        
        self.lstm = nn.LSTM(
            input_size=128 * 8, # frequency reduced to 8
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 1, 128, time]
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = F.relu(x + self.layer1(x))
        x = self.layer2(x) # [batch, 128, 8, time_reduced]
        
        batch, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(batch, t, c * f)
        
        lstm_out, _ = self.lstm(x)
        pooled = torch.mean(lstm_out, dim=1)
        return self.fc(pooled)


def build_model(config: Dict[str, Any]) -> nn.Module:
    """
    Builds and initializes model based on config parameters.
    """
    model_cfg = config.get("model", {})
    backbone = model_cfg.get("backbone", "cnn_lstm").lower()
    num_classes = model_cfg.get("num_classes", 9)
    dropout = model_cfg.get("dropout", 0.3)
    
    if backbone == "cnn_lstm":
        return DeepInfantCNNLSTM(
            in_channels=model_cfg.get("in_channels", 1),
            num_classes=num_classes,
            filters=model_cfg.get("cnn_filters", [32, 64, 128, 256]),
            lstm_hidden_dim=model_cfg.get("lstm_hidden_dim", 128),
            lstm_layers=model_cfg.get("lstm_layers", 2),
            bidirectional=model_cfg.get("bidirectional", True),
            dropout=dropout
        )
    elif backbone in ["resnet18_lstm", "resnet34_lstm"]:
        import torch.nn.functional as F
        return ResNetAudioLSTM(num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError(f"Unsupported model backbone: '{backbone}'")
