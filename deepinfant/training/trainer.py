"""
DeepInfant V2 Trainer Module.
Handles PyTorch training loop, validation, checkpointing, and scheduler step.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional

from deepinfant.training.losses import get_loss_function
from deepinfant.training.metrics import calculate_metrics, CLASSES


class Trainer:
    """Complete Training Manager for DeepInfant Models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        checkpoint_dir: str = "checkpoints"
    ):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.get("system", {}).get("device") == "cuda" else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 50)
        self.lr = train_cfg.get("learning_rate", 0.001)
        self.weight_decay = train_cfg.get("weight_decay", 0.0001)
        
        # Loss function
        loss_type = train_cfg.get("loss_function", "weighted_cross_entropy")
        self.criterion = get_loss_function(loss_type=loss_type, focal_gamma=train_cfg.get("focal_gamma", 2.0))
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3
        )
        
        self.best_val_f1 = 0.0

    def train_epoch(self) -> float:
        """Executes one training epoch."""
        self.model.train()
        total_loss = 0.0
        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * x.size(0)
        return total_loss / len(self.train_loader.dataset)

    def evaluate(self, loader: DataLoader) -> Tuple[float, Dict[str, Any]]:
        """Evaluates model on validation or test set."""
        self.model.eval()
        total_loss = 0.0
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                total_loss += loss.item() * x.size(0)
                
                preds = torch.argmax(logits, dim=1)
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                
        avg_loss = total_loss / len(loader.dataset)
        metrics = calculate_metrics(np.array(y_true), np.array(y_pred), CLASSES)
        return avg_loss, metrics

    def save_checkpoint(self, filename: str, epoch: int, val_metrics: Dict[str, Any]):
        """Saves complete checkpoint state including weights, optimizer, and metadata."""
        filepath = os.path.join(self.checkpoint_dir, filename)
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_val_f1": self.best_val_f1,
            "val_metrics": val_metrics,
            "config": self.config,
            "classes": CLASSES
        }
        torch.save(state, filepath)

    def fit(self) -> Dict[str, Any]:
        """Runs full training pipeline with checkpointing and early stopping."""
        history = []
        patience_counter = 0
        early_stopping_patience = self.config.get("training", {}).get("early_stopping_patience", 8)

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch()
            val_loss, val_metrics = self.evaluate(self.val_loader)
            val_f1 = val_metrics["macro_f1"]
            
            self.scheduler.step(val_f1)
            
            # Checkpoint best model
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.save_checkpoint("best_model.pt", epoch, val_metrics)
                patience_counter = 0
            else:
                patience_counter += 1

            self.save_checkpoint("last_model.pt", epoch, val_metrics)
            
            epoch_summary = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_f1
            }
            history.append(epoch_summary)

            if patience_counter >= early_stopping_patience:
                break
                
        return {"history": history, "best_val_f1": self.best_val_f1}
