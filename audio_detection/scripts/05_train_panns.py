#!/usr/bin/env python3
"""
Script 05b: Train PANNs CNN14

Fine-tunes PANNs CNN14 (pretrained on AudioSet) for drone detection.

Strategy:
    Phase 1 — Frozen backbone: only the 2-class head trains.
               LR = 3e-4 (lower than default to avoid head oscillation).
               Runs for up to 15 epochs with early stopping at patience=5.

    Phase 2 — Full fine-tune: backbone unfrozen at a very low LR (5e-5).
               Loads Phase 1 best checkpoint before starting.
               Runs for up to 30 epochs with early stopping at patience=7.

The lower LRs vs the CNN baseline are intentional: the pretrained weights
contain useful AudioSet knowledge that we want to preserve and shift
gradually, not overwrite.

Usage:
    python scripts/05_train_panns.py
    python scripts/05_train_panns.py --config configs/default.yaml
    python scripts/05_train_panns.py --phase2-only   # skip Phase 1, go straight to Phase 2
"""

import argparse
import copy
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.augmentation import AugmentationPipeline
from src.data.waveform_dataset import DroneWaveformDataset
from src.models.panns_cnn14 import PANNSCNN14
from src.training.trainer import Trainer
from src.utils.audio import load_background_clips
from src.utils.config import load_config


def make_loader(dataset, cfg, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        num_workers=cfg.training.num_workers,
        pin_memory=False,
    )


def build_augmentation(cfg, background_clips) -> AugmentationPipeline:
    return AugmentationPipeline(
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


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    splits_dir = Path(cfg.data.splits_dir)
    checkpoint_path = Path(cfg.outputs.checkpoints_dir) / "panns_cnn14" / "best.pt"

    print("=" * 60)
    print("UAV Audio Detection — Train PANNs CNN14")
    print("=" * 60)

    for split in ["train", "val"]:
        path = splits_dir / f"{split}.csv"
        if not path.exists():
            print(f"ERROR: {path} not found. Run scripts 02-04 first.")
            sys.exit(1)

    background_clips = load_background_clips(cfg.data.backgrounds_dir)
    if not background_clips:
        print("WARNING: data/backgrounds/ is empty. Background mixing disabled.")
    else:
        print(f"Loaded {len(background_clips)} background clips.")

    augmentation = build_augmentation(cfg, background_clips)

    train_dataset = DroneWaveformDataset.from_manifest(
        str(splits_dir / "train.csv"), split="train", augmentation=augmentation,
    )
    val_dataset = DroneWaveformDataset.from_manifest(
        str(splits_dir / "val.csv"), split="val",
    )

    print(f"\nTraining set:   {len(train_dataset):,} windows")
    print(f"Validation set: {len(val_dataset):,} windows")
    print(f"Class weights:  {train_dataset.class_weights.tolist()}")

    train_loader = make_loader(train_dataset, cfg, shuffle=True)
    val_loader = make_loader(val_dataset, cfg, shuffle=False)

    # ------------------------------------------------------------------ #
    # Phase 1: Train head only (backbone frozen)
    # ------------------------------------------------------------------ #
    if not args.phase2_only:
        print("\n--- Phase 1: Head-only training (backbone frozen) ---")
        print("    LR = 3e-4  |  max epochs = 15  |  patience = 5")

        model = PANNSCNN14(pretrained=True, freeze_backbone=True)

        cfg_phase1 = copy.deepcopy(cfg)
        cfg_phase1.training.learning_rate = 3e-4
        cfg_phase1.training.epochs = 15
        cfg_phase1.training.patience = 5

        trainer = Trainer(model, train_loader, val_loader, cfg_phase1)
        trainer.train()
        print(f"Phase 1 complete. Best checkpoint: {checkpoint_path}")
    else:
        print("\n--- Skipping Phase 1 (--phase2-only flag set) ---")
        if not checkpoint_path.exists():
            print(f"ERROR: No Phase 1 checkpoint found at {checkpoint_path}.")
            print("Run Phase 1 first (without --phase2-only).")
            sys.exit(1)

    # ------------------------------------------------------------------ #
    # Phase 2: Full fine-tune (backbone unfrozen, very low LR)
    # ------------------------------------------------------------------ #
    print("\n--- Phase 2: Full fine-tune (backbone unfrozen) ---")
    print("    LR = 5e-5  |  max epochs = 30  |  patience = 7")

    model = PANNSCNN14(pretrained=False, freeze_backbone=False)
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded Phase 1 weights from {checkpoint_path}")
    print("PANNs backbone unfrozen. Full model will train.")

    cfg_phase2 = copy.deepcopy(cfg)
    cfg_phase2.training.learning_rate = 5e-5
    cfg_phase2.training.epochs = 30
    cfg_phase2.training.patience = 7

    trainer2 = Trainer(model, train_loader, val_loader, cfg_phase2)
    trainer2.train()

    print(f"\nDone. Best checkpoint: {checkpoint_path}")
    print("NEXT STEP: python scripts/06_evaluate.py --checkpoint outputs/checkpoints/panns_cnn14/best.pt --model panns")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune PANNs CNN14 for drone detection.")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--phase2-only", action="store_true",
        help="Skip Phase 1 and go straight to Phase 2 (requires existing Phase 1 checkpoint).",
    )
    main(parser.parse_args())
