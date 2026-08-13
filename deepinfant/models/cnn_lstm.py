"""
DeepInfant V2 Core Architecture: Hybrid CNN + LSTM Model.
Extracts spatial/frequency features with 2D CNN backbone, models temporal progression
with Bidirectional LSTM, and outputs 9-class Softmax probability distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from deepinfant.models.cnn import CNNFeatureExtractor


class DeepInfantCNNLSTM(nn.Module):
    """
    Hybrid Deep Neural Network for Infant Cry Audio Classification.
    
    Architecture:
    Input: Log-Mel Spectrogram [batch, 1, 128, time_steps]
      ↓
    CNN Feature Extractor (4-Stage Conv2D + BatchNorm + MaxPool)
      ↓
    Feature Projection Layer
      ↓
    Bidirectional LSTM (2 Layers, 128 Hidden Dim)
      ↓
    Global Attention / Temporal Pooling
      ↓
    Dense + Dropout (0.3)
      ↓
    Softmax Classifier (9 classes)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 9,
        filters: list = None,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        
        if filters is None:
            filters = [32, 64, 128, 256]
            
        self.cnn = CNNFeatureExtractor(in_channels=in_channels, filters=filters)
        
        # Reduced frequency dimension is 128 // 8 = 16
        cnn_feature_dim = filters[3] * 16
        
        # Linear projection to reduce LSTM input dimension
        self.projection = nn.Sequential(
            nn.Linear(cnn_feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Temporal Modeling via LSTM
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        
        lstm_out_dim = lstm_hidden_dim * (2 if bidirectional else 1)
        
        # Dense Classification Head
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, 1, 128, time_steps]
        cnn_features = self.cnn(x) # [batch, time_reduced, cnn_feature_dim]
        
        proj_features = self.projection(cnn_features) # [batch, time_reduced, 256]
        
        lstm_out, _ = self.lstm(proj_features) # [batch, time_reduced, lstm_out_dim]
        
        # Temporal Average Pooling over time steps
        pooled = torch.mean(lstm_out, dim=1) # [batch, lstm_out_dim]
        
        logits = self.fc(pooled) # [batch, 9]
        return logits

    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities [batch, 9]."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
