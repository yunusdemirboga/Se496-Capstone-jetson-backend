#!/usr/bin/env python3
"""
Script 06: Evaluate

Runs full multi-tier evaluation on a trained model checkpoint.

Evaluation tiers:
  Tier 1 — In-distribution:    test split from data/splits/test.csv
  Tier 2 — (placeholder):      OOD evaluation requires identifying
                                acoustically distant test clusters
  Tier 3 — SNR sweep:          programmatic noise mixing at -10 to +20 dB
                                (requires data/backgrounds/ to be populated)

Results are saved as a JSON report in outputs/reports/.

Usage:
    python scripts/06_evaluate.py --checkpoint outputs/checkpoints/<model>/best.pt
    python scripts/06_evaluate.py --checkpoint outputs/checkpoints/<model>/best.pt --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import DroneAudioDataset
from src.evaluation.evaluator import Evaluator
from src.evaluation.report import plot_snr_curve, print_summary, save_report
from src.models.cnn_baseline import CNNBaseline
from src.utils.audio import load_background_clips
from src.utils.config import load_config


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)

    print("=" * 60)
    print("UAV Audio Detection — Evaluate")
    print("=" * 60)

    # Load model
    model = CNNBaseline()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    evaluator = Evaluator(model, cfg)
    results = {}

    # ------------------------------------------------------------------ #
    # Tier 1: In-distribution
    # ------------------------------------------------------------------ #
    print("\nTier 1: In-distribution evaluation...")
    feature_kwargs = dict(
        n_fft=cfg.data.n_fft,
        hop_length=cfg.data.hop_length,
        n_mels=cfg.data.n_mels,
        f_min=cfg.data.f_min,
        f_max=cfg.data.f_max,
        top_db=cfg.data.top_db,
    )
    test_dataset = DroneAudioDataset.from_manifest(
        str(Path(cfg.data.splits_dir) / "test.csv"),
        split="test",
        feature_type=cfg.data.feature_type,
        augmentation=None,
        sample_rate=cfg.data.sample_rate,
        **feature_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
    )
    results["tier1_in_distribution"] = evaluator.evaluate_tier1(test_loader)

    # ------------------------------------------------------------------ #
    # Tier 3: SNR sweep
    # ------------------------------------------------------------------ #
    background_clips = load_background_clips(cfg.data.backgrounds_dir)
    if background_clips:
        print("Tier 3: SNR sweep...")
        # Sample test windows for SNR evaluation
        test_manifest = test_dataset.manifest
        drone_paths = test_manifest[test_manifest["label"] == 1]["file_path"].tolist()
        nondrone_paths = test_manifest[test_manifest["label"] == 0]["file_path"].tolist()

        # Limit to 500 samples per class for reasonable runtime
        import random
        drone_waveforms = [np.load(p) for p in random.sample(drone_paths, min(500, len(drone_paths)))]
        nondrone_waveforms = [np.load(p) for p in random.sample(nondrone_paths, min(500, len(nondrone_paths)))]

        results["tier3_snr_sweep"] = evaluator.evaluate_snr_sweep(
            drone_waveforms=drone_waveforms,
            nondrone_waveforms=nondrone_waveforms,
            background_clips=background_clips,
            snr_levels_db=cfg.evaluation.snr_levels_db,
        )

        # Plot SNR curve
        snr_plot_path = Path(cfg.outputs.reports_dir) / f"{model.name}_snr_curve.png"
        plot_snr_curve(results["tier3_snr_sweep"], str(snr_plot_path), model_name=model.name)
        print(f"  SNR curve saved: {snr_plot_path}")
    else:
        print("Tier 3 skipped: data/backgrounds/ is empty.")

    # ------------------------------------------------------------------ #
    # Print and save
    # ------------------------------------------------------------------ #
    print_summary(results, model_name=model.name)
    report_path = save_report(results, model_name=model.name, reports_dir=cfg.outputs.reports_dir)
    print(f"\nReport saved: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained drone detection model.")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint (best.pt).",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config file.",
    )
    main(parser.parse_args())
