"""
Evaluator: runs the model across all evaluation tiers and collects results.

Tiers:
  1. In-distribution  — test split from data/splits/test.csv
  2. Out-of-distribution — acoustically distant test clusters (subset of tier 1)
  3. SNR sweep — programmatic noise mixing at specified dB levels
  (Tier 4 / real-world is handled separately with manually collected data)

See docs/evaluation_protocol.md for tier definitions and target metrics.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.augmentation import mix_with_background
from src.evaluation.metrics import compute_metrics, compute_snr_sweep_metrics
from src.models.base import DroneDetectionModel
from src.utils.config import Config


class Evaluator:
    """
    Runs inference and computes metrics for a trained model.

    Args:
        model:   Trained DroneDetectionModel.
        config:  Project configuration.
        device:  Torch device for inference.
    """

    def __init__(
        self,
        model: DroneDetectionModel,
        config: Config,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _run_inference(
        self, loader: DataLoader
    ) -> tuple:
        """Run inference over a DataLoader. Returns (y_true, y_pred, y_prob)."""
        all_true, all_pred, all_prob = [], [], []
        threshold = self.config.evaluation.threshold

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            logits = self.model(batch_x)
            probs = torch.softmax(logits, dim=-1)[:, 1]  # P(drone)

            preds = (probs >= threshold).long()

            all_true.append(batch_y.numpy())
            all_pred.append(preds.cpu().numpy())
            all_prob.append(probs.cpu().numpy())

        return (
            np.concatenate(all_true),
            np.concatenate(all_pred),
            np.concatenate(all_prob),
        )

    def evaluate_tier1(self, test_loader: DataLoader) -> Dict:
        """
        Tier 1: In-distribution evaluation on the held-out test split.

        High performance here is expected and is a sanity check only.
        It does not imply deployability.
        """
        y_true, y_pred, y_prob = self._run_inference(test_loader)
        return compute_metrics(
            y_true, y_pred, y_prob,
            fpr_operating_point=self.config.evaluation.fpr_operating_point,
            threshold=self.config.evaluation.threshold,
        )

    def evaluate_snr_sweep(
        self,
        drone_waveforms: List[np.ndarray],
        nondrone_waveforms: List[np.ndarray],
        background_clips: List[np.ndarray],
        snr_levels_db: Optional[List[float]] = None,
    ) -> Dict[float, Dict]:
        """
        Tier 3: Evaluate at multiple SNR levels.

        Drone clips are mixed with background sounds at each SNR level.
        Non-drone clips are evaluated without mixing (they define the
        false positive rate at each noise condition).

        Args:
            drone_waveforms:    List of clean drone windows (1s, float32).
            nondrone_waveforms: List of clean non-drone windows.
            background_clips:   Background sounds for mixing.
            snr_levels_db:      List of SNR levels to evaluate. Defaults to
                                config.evaluation.snr_levels_db.

        Returns:
            {snr_db: metrics_dict}
        """
        from src.data.features import extract_features

        if snr_levels_db is None:
            snr_levels_db = self.config.evaluation.snr_levels_db

        import random

        y_true_by_snr: Dict[float, np.ndarray] = {}
        y_pred_by_snr: Dict[float, np.ndarray] = {}
        y_prob_by_snr: Dict[float, np.ndarray] = {}

        for snr in snr_levels_db:
            all_true, all_pred, all_prob = [], [], []

            # Drone clips mixed with background at this SNR
            for waveform in drone_waveforms:
                background = random.choice(background_clips)
                mixed = mix_with_background(waveform, background, snr_db=snr)
                features = extract_features(
                    mixed,
                    feature_type=self.config.data.feature_type,
                    sample_rate=self.config.data.sample_rate,
                )
                x = torch.from_numpy(features).float().unsqueeze(0).unsqueeze(0).to(self.device)
                logits = self.model(x)
                prob = float(torch.softmax(logits, dim=-1)[0, 1].cpu())
                pred = int(prob >= self.config.evaluation.threshold)
                all_true.append(1)
                all_pred.append(pred)
                all_prob.append(prob)

            # Non-drone clips (no mixing — these define FPR)
            for waveform in nondrone_waveforms:
                features = extract_features(
                    waveform,
                    feature_type=self.config.data.feature_type,
                    sample_rate=self.config.data.sample_rate,
                )
                x = torch.from_numpy(features).float().unsqueeze(0).unsqueeze(0).to(self.device)
                logits = self.model(x)
                prob = float(torch.softmax(logits, dim=-1)[0, 1].cpu())
                pred = int(prob >= self.config.evaluation.threshold)
                all_true.append(0)
                all_pred.append(pred)
                all_prob.append(prob)

            y_true_by_snr[snr] = np.array(all_true)
            y_pred_by_snr[snr] = np.array(all_pred)
            y_prob_by_snr[snr] = np.array(all_prob)

        return compute_snr_sweep_metrics(
            y_true_by_snr,
            y_pred_by_snr,
            y_prob_by_snr,
            fpr_operating_point=self.config.evaluation.fpr_operating_point,
        )
