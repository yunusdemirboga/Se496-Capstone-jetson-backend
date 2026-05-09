"""
Loss functions for imbalanced binary audio classification.

After windowing, the class distribution is approximately balanced
(~1:1.4 drone:non-drone). Class-weighted cross-entropy is the default
and handles any residual imbalance. Focal loss is available for cases
where the model struggles with hard negatives.

See DESIGN.md §2.3 for the class imbalance mitigation strategy.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with per-class weighting.

    Weights are typically computed from the training set distribution:
        weight[c] = n_total / (n_classes * n_samples_in_class_c)

    These are available via DroneAudioDataset.class_weights.

    Args:
        class_weights:    Tensor of shape (n_classes,). Pass None for uniform weights.
        label_smoothing:  Smoothing factor in [0, 1]. Reduces overconfidence.
                          Recommended: 0.05.
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )


class FocalLoss(nn.Module):
    """
    Focal loss for hard example mining.

    Downweights easy, well-classified examples so the model focuses
    on harder ones. Most useful when class-weighted CE still struggles
    with specific failure modes.

    Reference: Lin et al., "Focal Loss for Dense Object Detection", 2017.

    Args:
        alpha: Per-class weight tensor of shape (n_classes,).
               Equivalent to class_weights in WeightedCrossEntropyLoss.
        gamma: Focusing parameter. gamma=0 reduces to cross-entropy.
               Common values: 1.0, 2.0. Higher = more focus on hard examples.
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits, targets, weight=self.alpha, reduction="none"
        )
        p_t = torch.exp(-ce_loss)
        focal_weight = (1.0 - p_t) ** self.gamma
        return (focal_weight * ce_loss).mean()
