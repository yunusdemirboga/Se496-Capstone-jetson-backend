"""
Training loop for drone audio detection models.

Handles:
  - Epoch-level training and validation
  - Class-weighted loss (from dataset.class_weights)
  - Gradient clipping
  - Learning rate scheduling (ReduceLROnPlateau)
  - Early stopping
  - Checkpoint saving (best validation loss)
  - Training history logging

Usage:
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()
    # Best checkpoint saved to outputs/checkpoints/<model_name>/best.pt
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from src.models.base import DroneDetectionModel
from src.training.losses import WeightedCrossEntropyLoss
from src.utils.config import Config


class Trainer:
    """
    Training loop with early stopping and automatic checkpointing.

    Args:
        model:        DroneDetectionModel instance.
        train_loader: DataLoader for the training split.
        val_loader:   DataLoader for the validation split.
        config:       Project configuration (configs/default.yaml).
        device:       Torch device. Defaults to CUDA if available, else CPU.
    """

    def __init__(
        self,
        model: DroneDetectionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Config,
        device: Optional[torch.device] = None,
        resume_checkpoint: Optional[str] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )

        class_weights = train_loader.dataset.class_weights.to(self.device)
        self.criterion = WeightedCrossEntropyLoss(
            class_weights=class_weights,
            label_smoothing=config.training.label_smoothing,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=3, factor=0.5
        )

        self.checkpoint_dir = Path(config.outputs.checkpoints_dir) / model.name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = []
        self.start_epoch = 1

        if resume_checkpoint:
            self._load_checkpoint(resume_checkpoint)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model and optimizer state from a checkpoint to resume training."""
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_val_loss = ckpt["best_val_loss"]
        self.start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from {checkpoint_path}")
        print(f"  Best val loss so far: {self.best_val_loss:.4f} (epoch {ckpt['epoch']})")
        print(f"  Resuming from epoch {self.start_epoch}")

    def train_epoch(self) -> float:
        """Run one training epoch. Returns mean loss over all batches."""
        self.model.train()
        total_loss = 0.0

        for batch_x, batch_y in self.train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def val_epoch(self) -> float:
        """Run one validation epoch. Returns mean loss over all batches."""
        self.model.eval()
        total_loss = 0.0

        for batch_x, batch_y in self.val_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)
            total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self) -> None:
        """
        Run full training with early stopping.

        Saves the best checkpoint to:
            outputs/checkpoints/<model_name>/best.pt
        """
        params = self.model.parameter_count()
        print(f"\nTraining: {self.model.name}")
        print(f"Device:   {self.device}")
        print(f"Params:   {params['trainable']:,} trainable / {params['total']:,} total")
        print(f"Epochs:   {self.config.training.epochs} (patience={self.config.training.patience})")
        print("-" * 60)

        for epoch in range(self.start_epoch, self.config.training.epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.val_epoch()
            self.scheduler.step(val_loss)

            entry = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
            }
            self.history.append(entry)

            improved = val_loss < self.best_val_loss
            marker = " *" if improved else ""
            print(
                f"Epoch {epoch:3d}/{self.config.training.epochs} | "
                f"train={train_loss:.4f} | val={val_loss:.4f}{marker}"
            )

            if improved:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint("best.pt")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.training.patience:
                print(f"\nEarly stopping triggered at epoch {epoch}.")
                break

        self._save_history()
        print(f"\nBest validation loss: {self.best_val_loss:.4f}")
        print(f"Checkpoint: {self.checkpoint_dir / 'best.pt'}")

    def _save_checkpoint(self, filename: str) -> None:
        torch.save(
            {
                "model_name": self.model.name,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
                "epoch": len(self.history),
            },
            self.checkpoint_dir / filename,
        )

    def _save_history(self) -> None:
        path = self.checkpoint_dir / "training_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
