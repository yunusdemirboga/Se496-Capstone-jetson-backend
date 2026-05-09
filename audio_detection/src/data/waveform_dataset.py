"""
Waveform dataset for PANNs-style models.

Returns raw waveforms instead of pre-computed spectrograms, since
PANNs CNN14 handles spectrogram extraction internally.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.augmentation import AugmentationPipeline


class DroneWaveformDataset(Dataset):
    """
    Dataset that returns raw waveforms for PANNs-style models.

    Loads .npy window files and applies optional waveform augmentation.
    Feature extraction is handled by the model itself.

    Args:
        manifest:    DataFrame with clip_id, file_path, label, split columns.
        augmentation: Optional AugmentationPipeline (waveform-level).
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        augmentation: Optional[AugmentationPipeline] = None,
    ):
        self.manifest = manifest.reset_index(drop=True)
        self.augmentation = augmentation

        n_total = len(self.manifest)
        n_drone = int((self.manifest["label"] == 1).sum())
        n_nondrone = int((self.manifest["label"] == 0).sum())

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

        waveform_tensor = torch.from_numpy(waveform).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        return waveform_tensor, label_tensor

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str,
        split: str,
        **kwargs,
    ) -> "DroneWaveformDataset":
        df = pd.read_csv(manifest_path)
        split_df = df[df["split"] == split].copy()
        if split_df.empty:
            raise ValueError(
                f"No rows found for split='{split}' in {manifest_path}."
            )
        return cls(split_df, **kwargs)
