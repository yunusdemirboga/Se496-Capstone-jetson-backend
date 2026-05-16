#!/usr/bin/env python3
"""
Script 05: Train

Trains a drone detection model using the prepared windows and splits.

Usage:
    python scripts/05_train.py [--config configs/default.yaml]
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.augmentation import AugmentationPipeline
from src.data.dataset import DroneAudioDataset
from src.data.spec_augment import FreqWarp, SpecAugment
from src.models.cnn_baseline import CNNBaseline
from src.training.trainer import Trainer
from src.utils.audio import load_background_clips
from src.utils.config import load_config


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    splits_dir = Path(cfg.data.splits_dir)

    print("=" * 60)
    print("UAV Audio Detection — Train")
    print("=" * 60)

    for split in ["train", "val"]:
        path = splits_dir / f"{split}.csv"
        if not path.exists():
            print(f"ERROR: {path} not found. Run scripts 02-04 first.")
            sys.exit(1)

    # Load background clips for augmentation
    background_clips = []
    if cfg.augmentation.enabled:
        background_clips = load_background_clips(cfg.data.backgrounds_dir)
        if not background_clips:
            print(
                "WARNING: data/backgrounds/ is empty. Background mixing augmentation "
                "will be disabled.\n"
                "Download background sounds (ESC-50, DEMAND, etc.) and place them "
                f"in {cfg.data.backgrounds_dir} for best robustness."
            )
        else:
            print(f"Loaded {len(background_clips)} background clips for augmentation.")

    augmentation = None
    spec_augment = None
    freq_warp = None
    if cfg.augmentation.enabled:
        augmentation = AugmentationPipeline(
            background_clips=background_clips,
            snr_range_db=tuple(cfg.augmentation.snr_range_db),
            background_mix_prob=cfg.augmentation.background_mix_prob,
            noise_prob=cfg.augmentation.noise_prob,
            noise_snr_range_db=tuple(cfg.augmentation.noise_snr_range_db),
            pitch_shift_prob=cfg.augmentation.pitch_shift_prob,
            pitch_shift_range=tuple(cfg.augmentation.pitch_shift_range_semitones),
            time_stretch_prob=cfg.augmentation.time_stretch_prob,
            time_stretch_range=tuple(cfg.augmentation.time_stretch_range),
            rir_prob=cfg.augmentation.rir_prob,
            rir_rt60_range_sec=tuple(cfg.augmentation.rir_rt60_range_sec),
            gain_prob=cfg.augmentation.gain_prob,
            gain_range_db=tuple(cfg.augmentation.gain_range_db),
            sample_rate=cfg.data.sample_rate,
        )
        spec_augment = SpecAugment(
            freq_mask_param=cfg.augmentation.freq_mask_param,
            time_mask_param=cfg.augmentation.time_mask_param,
            num_freq_masks=cfg.augmentation.num_freq_masks,
            num_time_masks=cfg.augmentation.num_time_masks,
            prob=cfg.augmentation.spec_augment_prob,
        )
        freq_warp = FreqWarp(
            max_warp=cfg.augmentation.freq_warp_max,
            prob=cfg.augmentation.freq_warp_prob,
        )

    feature_kwargs = dict(
        n_fft=cfg.data.n_fft,
        hop_length=cfg.data.hop_length,
        n_mels=cfg.data.n_mels,
        f_min=cfg.data.f_min,
        f_max=cfg.data.f_max,
        top_db=cfg.data.top_db,
    )

    train_dataset = DroneAudioDataset.from_manifest(
        str(splits_dir / "train.csv"),
        split="train",
        feature_type=cfg.data.feature_type,
        augmentation=augmentation,
        spec_augment=spec_augment,
        freq_warp=freq_warp,
        sample_rate=cfg.data.sample_rate,
        **feature_kwargs,
    )
    val_dataset = DroneAudioDataset.from_manifest(
        str(splits_dir / "val.csv"),
        split="val",
        feature_type=cfg.data.feature_type,
        augmentation=None,
        spec_augment=None,
        freq_warp=None,
        sample_rate=cfg.data.sample_rate,
        **feature_kwargs,
    )

    print(f"\nTraining set:   {len(train_dataset):,} windows")
    print(f"Validation set: {len(val_dataset):,} windows")
    print(f"Class weights:  {train_dataset.class_weights.tolist()}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
    )

    model = CNNBaseline()

    trainer = Trainer(model, train_loader, val_loader, cfg,
                      resume_checkpoint=args.resume)
    trainer.train()

    print(f"\nNEXT STEP: python scripts/06_evaluate.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a drone audio detection model.")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to a checkpoint (.pt) to resume training from.",
    )
    main(parser.parse_args())
