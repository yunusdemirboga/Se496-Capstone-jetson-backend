"""
PyTorch Dataset for UAV audio detection.

Loads fixed-length windows from disk (.npy files), applies optional
augmentation, and computes features on-the-fly.

Features are computed per-sample rather than pre-computed to allow
experimenting with different feature types without re-running the
windowing pipeline. For large-scale training, consider pre-computing
features and caching them.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.augmentation import AugmentationPipeline
from src.data.features import FeatureType, extract_features
from src.data.spec_augment import FreqWarp, SpecAugment


class DroneAudioDataset(Dataset):
    """
    Dataset for drone audio detection from fixed-length windows.

    Expects a manifest CSV with columns:
        clip_id   — unique identifier
        file_path — absolute path to the .npy window file
        label     — 0 (no drone) or 1 (drone)
        split     — "train", "val", or "test"

    The class_weights property provides per-class weights for use in a
    weighted loss function. Weights are computed from the manifest and
    correct for any remaining class imbalance after windowing.

    Args:
        manifest:      DataFrame of the subset to load (already filtered by split).
        feature_type:  Which feature representation to use.
        augmentation:  Optional AugmentationPipeline (apply during training only).
        sample_rate:   Audio sample rate.
        feature_kwargs: Additional kwargs forwarded to the feature extractor.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        feature_type: FeatureType = "log_mel",
        augmentation: Optional[AugmentationPipeline] = None,
        spec_augment: Optional[SpecAugment] = None,
        freq_warp: Optional[FreqWarp] = None,
        sample_rate: int = 16000,
        **feature_kwargs,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.feature_type = feature_type
        self.augmentation = augmentation
        self.spec_augment = spec_augment
        self.freq_warp = freq_warp
        self.sample_rate = sample_rate
        self.feature_kwargs = feature_kwargs

        n_total = len(self.manifest)
        n_drone = int((self.manifest["label"] == 1).sum())
        n_nondrone = int((self.manifest["label"] == 0).sum())

        # Inverse frequency weighting — avoids division by zero
        w_nondrone = n_total / (2 * max(n_nondrone, 1))
        w_drone = n_total / (2 * max(n_drone, 1))
        self.class_weights = torch.tensor([w_nondrone, w_drone], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[idx]

        waveform = np.load(row["file_path"])
        label = int(row["label"])

        if self.augmentation is not None:
            waveform = self.augmentation(waveform, label=label)

        features = extract_features(
            waveform,
            feature_type=self.feature_type,
            sample_rate=self.sample_rate,
            **self.feature_kwargs,
        )

        if self.freq_warp is not None:
            features = self.freq_warp(features)

        if self.spec_augment is not None:
            features = self.spec_augment(features)

        # Add channel dim: (freq_bins, time) → (1, freq_bins, time)
        features_tensor = torch.from_numpy(features).float().unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return features_tensor, label_tensor

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str,
        split: str,
        **kwargs,
    ) -> "DroneAudioDataset":
        """
        Convenience constructor: load a specific split from a manifest CSV.

        Args:
            manifest_path: Path to the split CSV (e.g., data/splits/train.csv).
            split:         "train", "val", or "test".
            **kwargs:      Forwarded to __init__.
        """
        df = pd.read_csv(manifest_path)
        split_df = df[df["split"] == split].copy()
        if split_df.empty:
            raise ValueError(
                f"No rows found for split='{split}' in {manifest_path}. "
                f"Available splits: {df['split'].unique().tolist()}"
            )
        return cls(split_df, **kwargs)
