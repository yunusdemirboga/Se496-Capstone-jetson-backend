"""
Abstract base class for all drone detection models.

All model implementations inherit from DroneDetectionModel and implement
forward(). This ensures a consistent interface for training, evaluation,
and inference regardless of architecture.

Input convention:
    Batch of log-mel spectrograms: (batch, 1, freq_bins, time_frames)
    For the default config: (batch, 1, 128, 101)

Output convention:
    Logits: (batch, 2)  — class 0=no drone, class 1=drone
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import torch
import torch.nn as nn


class DroneDetectionModel(nn.Module, ABC):
    """
    Abstract base for all drone detection model architectures.

    Subclasses must implement:
        forward(x) -> logits
        name (property) -> str
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 1, freq_bins, time_frames).

        Returns:
            Logits of shape (batch, 2).
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model identifier (used for checkpoint naming)."""
        ...

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get class predictions and probabilities from a batch.

        Args:
            x: Input tensor.

        Returns:
            (predictions, probabilities):
                predictions  — (batch,) int64 tensor of predicted class indices
                probabilities — (batch, 2) float tensor of class probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        return preds, probs

    def parameter_count(self) -> Dict[str, int]:
        """Return total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
