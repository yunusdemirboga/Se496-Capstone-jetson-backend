"""
PANNs CNN14 — fine-tuned for drone detection.

Pretrained on AudioSet (2 million YouTube clips, 527 sound classes).
Reference: Kong et al., "PANNs: Large-Scale Pretrained Audio Neural
Networks for Audio Pattern Recognition", IEEE/ACM TASLP 2020.

Weights: https://zenodo.org/record/3987831 (Cnn14_16k_mAP=0.438.pth)

Architecture:
    Input: raw waveform (batch, 16000) at 16kHz
    Internal: log-mel spectrogram (64 mel bins, matched to pretrained weights)
    Backbone: 6 convolutional blocks (frozen by default)
    Head: Linear(2048, 2) trained from scratch for drone/no-drone

Usage:
    model = PANNSCNN14(pretrained=True, freeze_backbone=True)
    model.unfreeze_backbone()  # call after head has converged
"""

import ssl
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from src.models.base import DroneDetectionModel

WEIGHTS_URL = (
    "https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth"
)
WEIGHTS_CACHE = Path.home() / ".cache" / "panns" / "Cnn14_16k.pth"


def _download_weights() -> Path:
    if WEIGHTS_CACHE.exists():
        return WEIGHTS_CACHE
    WEIGHTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading PANNs CNN14 pretrained weights (~80 MB)...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(WEIGHTS_URL, context=ctx) as response, \
            open(WEIGHTS_CACHE, "wb") as f:
        f.write(response.read())
    print(f"Saved to {WEIGHTS_CACHE}")
    return WEIGHTS_CACHE


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, pool_size=(2, 2)) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, pool_size)


class PANNSCNN14(DroneDetectionModel):
    """
    CNN14 backbone from PANNs, fine-tuned for drone/no-drone detection.

    The backbone (bn0 + 6 conv blocks + fc1) is initialized from pretrained
    AudioSet weights and frozen by default. Only the 2-class classifier head
    is trained initially. Call unfreeze_backbone() to fine-tune the full network
    at a lower learning rate after the head has converged.

    Args:
        pretrained:       Load pretrained AudioSet weights (recommended: True).
        freeze_backbone:  Freeze backbone during initial training.
        dropout:          Dropout rate in the backbone (matches original PANNs).
    """

    # Spectrogram parameters — must match the pretrained checkpoint exactly.
    SAMPLE_RATE = 16000
    N_FFT = 512
    HOP_LENGTH = 160
    N_MELS = 64
    F_MIN = 50.0
    F_MAX = 8000.0

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        self._dropout = dropout

        # Spectrogram extraction (not part of checkpoint — handled here via torchaudio)
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.SAMPLE_RATE,
            n_fft=self.N_FFT,
            hop_length=self.HOP_LENGTH,
            n_mels=self.N_MELS,
            f_min=self.F_MIN,
            f_max=self.F_MAX,
            window_fn=torch.hann_window,
            power=2.0,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=80.0
        )

        # Backbone — names match pretrained checkpoint keys exactly
        self.bn0 = nn.BatchNorm2d(self.N_MELS)
        self.conv_block1 = ConvBlock(1, 64)
        self.conv_block2 = ConvBlock(64, 128)
        self.conv_block3 = ConvBlock(128, 256)
        self.conv_block4 = ConvBlock(256, 512)
        self.conv_block5 = ConvBlock(512, 1024)
        self.conv_block6 = ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048, bias=True)

        # 2-class head (replaces PANNs 527-class fc_audioset)
        self.classifier = nn.Linear(2048, 2, bias=True)

        if pretrained:
            self._load_pretrained()

        if freeze_backbone:
            self.freeze_backbone()

    def _load_pretrained(self) -> None:
        weights_path = _download_weights()
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        pretrained_state = checkpoint["model"]
        own_state = self.state_dict()

        loaded = skipped = 0
        for key, param in pretrained_state.items():
            # Skip spectrogram extractor (we use torchaudio transforms instead)
            if key.startswith(("spectrogram_extractor", "logmel_extractor")):
                skipped += 1
                continue
            # Skip 527-class AudioSet head
            if key.startswith("fc_audioset"):
                skipped += 1
                continue
            if key in own_state and own_state[key].shape == param.shape:
                own_state[key].copy_(param)
                loaded += 1
            else:
                skipped += 1

        print(f"PANNs CNN14: loaded {loaded} weight tensors, skipped {skipped}.")

    def freeze_backbone(self) -> None:
        """
        Freeze the pretrained backbone — only the classifier head trains.

        Critically, frozen backbone modules are also set to eval() mode.
        This ensures BatchNorm uses the pretrained AudioSet running statistics
        (rather than per-batch statistics) consistently during both training
        and validation forward passes. Without this, BN behaves differently
        in train() vs eval() mode, causing train_loss << val_loss.
        """
        backbone_modules = [
            self.bn0, self.conv_block1, self.conv_block2, self.conv_block3,
            self.conv_block4, self.conv_block5, self.conv_block6, self.fc1,
        ]
        for module in backbone_modules:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
        print("PANNs backbone frozen (eval mode). Only classifier head will train.")

    def train(self, mode: bool = True):
        """
        Override train() to keep frozen backbone modules in eval() mode.

        PyTorch's default model.train() sets ALL submodules to training mode,
        which would undo our freeze_backbone() eval() call. We intercept this
        to keep frozen modules in eval() whenever the backbone is frozen.
        """
        super().train(mode)
        # If backbone is frozen (bn0 has no grads), re-apply eval to backbone
        if not self.bn0.weight.requires_grad:
            backbone_modules = [
                self.bn0, self.conv_block1, self.conv_block2, self.conv_block3,
                self.conv_block4, self.conv_block5, self.conv_block6, self.fc1,
            ]
            for module in backbone_modules:
                module.eval()
        return self

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        print("PANNs backbone unfrozen. Full model will train.")

    @property
    def name(self) -> str:
        return "panns_cnn14"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw waveform tensor of shape (batch, 16000).

        Returns:
            Logits of shape (batch, 2).
        """
        # Log-mel spectrogram: (batch, n_mels, time)
        x = self.melspec(x)
        x = self.amplitude_to_db(x)

        # PANNs format: (batch, 1, time, n_mels)
        x = x.unsqueeze(1).permute(0, 1, 3, 2)

        # Normalize over mel dimension (bn0 sees mel in channel position)
        x = x.transpose(1, 3)   # (batch, n_mels, time, 1)
        x = self.bn0(x)
        x = x.transpose(1, 3)   # (batch, 1, time, n_mels)

        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block1(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1))
        x = F.dropout(x, p=0.2, training=self.training)

        # Global pooling: mean over mel, then max+mean over time
        x = torch.mean(x, dim=3)              # (batch, 2048, time)
        x = torch.max(x, dim=2)[0] + torch.mean(x, dim=2)  # (batch, 2048)

        x = F.dropout(x, p=self._dropout, training=self.training)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=self._dropout, training=self.training)

        return self.classifier(x)
