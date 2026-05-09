"""
CNN Baseline — log-mel spectrogram classifier.

Architecture: three convolutional blocks followed by global average
pooling and a two-layer classifier head.

Input:  (batch, 1, 128, 101)  — 1-channel log-mel spectrogram
Output: (batch, 2)            — logits for [no-drone, drone]

Each conv block:
    Conv2d → BatchNorm → ReLU → MaxPool2d

Global average pooling collapses the spatial dimensions so the
classifier head is independent of input size, which also reduces
the parameter count and acts as implicit regularization.
"""

import torch
import torch.nn as nn

from src.models.base import DroneDetectionModel


class CNNBaseline(DroneDetectionModel):
    """
    Three-block CNN for drone audio detection.

    Designed for log-mel spectrogram input of shape (batch, 1, 128, 101).
    Achieves a good tradeoff between capacity and training speed on CPU.
    """

    def __init__(self, dropout: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: (1, 128, 101) → (32, 64, 50)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: (32, 64, 50) → (64, 32, 25)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: (64, 32, 25) → (128, 16, 12)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global average pool: (128, H, W) → (128,)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    @property
    def name(self) -> str:
        return "cnn_baseline"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x)
        return self.classifier(x)
