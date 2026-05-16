#!/usr/bin/env python3
"""
Script 04: Verify Split Integrity

Checks that the generated splits are free of source-level leakage.

A split is clean if no source clip has windows in more than one split.
If leakage is detected, rerun script 03 with a different seed or
a larger number of clusters.

This script is fast (pure CSV operations, no audio I/O) and should
be run after every split generation.

Usage:
    python scripts/04_verify_splits.py [--config configs/default.yaml]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.splitting import verify_no_source_leakage
from src.utils.config import load_config


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    splits_dir = Path(cfg.data.splits_dir)

    all_windows_path = splits_dir / "all_windows.csv"
    if not all_windows_path.exists():
        print(f"ERROR: {all_windows_path} not found. Run 03_generate_splits.py first.")
        sys.exit(1)

    print("=" * 60)
    print("UAV Audio Detection — Verify Split Integrity")
    print("=" * 60)

    manifest = pd.read_csv(all_windows_path)
    print(f"Loaded: {len(manifest):,} windows from {all_windows_path.name}\n")

    # ------------------------------------------------------------------ #
    # Source leakage check
    # ------------------------------------------------------------------ #
    is_clean, report = verify_no_source_leakage(manifest, source_col="source_clip_idx")

    print("SOURCE LEAKAGE CHECK")
    print("-" * 40)
    print(f"  Total source clips:   {report['total_sources']:,}")
    print(f"  Leaking source clips: {report['leaking_sources']:,}")

    if is_clean:
        print(f"\n  PASSED — No source clip appears in multiple splits.")
        print(f"  Splits are clean and safe to use for training and evaluation.")
    else:
        print(f"\n  FAILED — {report['leaking_sources']} source clips appear in multiple splits.")
        print(f"  Leaking source IDs (first 20): {report['leaking_source_ids']}")
        print(f"\n  ACTION REQUIRED: Rerun 03_generate_splits.py with more clusters or a different seed.")

    # ------------------------------------------------------------------ #
    # Class balance per split
    # ------------------------------------------------------------------ #
    print("\nCLASS BALANCE PER SPLIT")
    print("-" * 40)
    for split_name in ["train", "val", "test"]:
        subset = manifest[manifest["split"] == split_name]
        if subset.empty:
            print(f"  {split_name}: (empty)")
            continue
        n_drone = int((subset["label"] == 1).sum())
        n_nondrone = int((subset["label"] == 0).sum())
        n_total = len(subset)
        print(f"  {split_name:5}: {n_total:>8,} windows | "
              f"drone={n_drone:,} ({100*n_drone/n_total:.1f}%) | "
              f"non-drone={n_nondrone:,} ({100*n_nondrone/n_total:.1f}%)")

    # ------------------------------------------------------------------ #
    # Overlap check between train and test at source level
    # ------------------------------------------------------------------ #
    print("\nTRAIN / TEST SOURCE OVERLAP CHECK")
    print("-" * 40)
    train_sources = set(manifest[manifest["split"] == "train"]["source_clip_idx"])
    val_sources = set(manifest[manifest["split"] == "val"]["source_clip_idx"])
    test_sources = set(manifest[manifest["split"] == "test"]["source_clip_idx"])

    train_test_overlap = train_sources & test_sources
    train_val_overlap = train_sources & val_sources
    val_test_overlap = val_sources & test_sources

    print(f"  Train ∩ Test: {len(train_test_overlap)} source clips")
    print(f"  Train ∩ Val:  {len(train_val_overlap)} source clips")
    print(f"  Val   ∩ Test: {len(val_test_overlap)} source clips")

    if train_test_overlap or train_val_overlap or val_test_overlap:
        print("\n  OVERLAP DETECTED — this indicates a bug in the split generation.")
        sys.exit(1)
    else:
        print("\n  No overlap detected. Splits are disjoint at source level.")

    print("\n" + "=" * 60)
    if is_clean:
        print("ALL CHECKS PASSED. Ready for training.")
        print("NEXT STEP: python scripts/05_train.py")
    else:
        print("CHECKS FAILED. Fix split generation before training.")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify train/val/test split integrity.")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config file.",
    )
    main(parser.parse_args())
