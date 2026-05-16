#!/usr/bin/env python3
"""
Script 03: Generate Source-Aware Splits

Generates train / val / test splits that respect source boundaries.

The DADS dataset has no source_id column. We approximate source groups
by clustering clips on acoustic fingerprints (spectral centroid,
bandwidth, rolloff, RMS, duration). Clips from the same recording
session tend to cluster together. Entire clusters are assigned to a
single split.

The split manifests are saved to data/splits/ and should be committed
to version control. Do not regenerate splits unless there is a specific
reason — doing so changes the evaluation baseline.

After running this script, run 04_verify_splits.py to confirm there
is no source-level leakage.

Usage:
    python scripts/03_generate_splits.py [--config configs/default.yaml]
"""

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.splitting import generate_splits
from src.utils.config import load_config


def compute_source_fingerprints(
    manifest: pd.DataFrame,
    sample_rate: int = 16000,
) -> pd.DataFrame:
    """
    Compute acoustic fingerprints for each SOURCE CLIP (not each window).

    We use one representative window per source clip (the first window)
    to compute the fingerprint. This is the basis for clustering.

    Returns:
        DataFrame with source_clip_idx, label, duration_sec, and spectral features.
    """
    # One representative window per source clip
    first_windows = (
        manifest.sort_values("clip_id")
        .groupby("source_clip_idx")
        .first()
        .reset_index()[["source_clip_idx", "label", "file_path"]]
    )

    records = []
    for _, row in tqdm(
        first_windows.iterrows(),
        total=len(first_windows),
        desc="Computing fingerprints",
    ):
        waveform = np.load(row["file_path"])
        duration_sec = len(waveform) / sample_rate

        centroid = librosa.feature.spectral_centroid(y=waveform, sr=sample_rate)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sample_rate)[0]
        rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sample_rate)[0]
        rms = librosa.feature.rms(y=waveform)[0]

        records.append(
            {
                "source_clip_idx": int(row["source_clip_idx"]),
                "label": int(row["label"]),
                "duration_sec": float(duration_sec),
                "spectral_centroid_mean": float(np.mean(centroid)),
                "spectral_bandwidth_mean": float(np.mean(bandwidth)),
                "spectral_rolloff_mean": float(np.mean(rolloff)),
                "rms_mean": float(np.mean(rms)),
            }
        )

    return pd.DataFrame(records)


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)

    processed_dir = Path(cfg.data.processed_dir)
    splits_dir = Path(cfg.data.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UAV Audio Detection — Generate Splits")
    print("=" * 60)
    print(f"Strategy:     Acoustic cluster-based (source-aware)")
    print(f"Clusters:     drone={cfg.data.n_clusters_drone}, non-drone={cfg.data.n_clusters_nondrone}")
    print(f"Split ratios: train={cfg.data.train_frac}, val={cfg.data.val_frac}, "
          f"test={1-cfg.data.train_frac-cfg.data.val_frac:.2f}")
    print(f"Seed:         {cfg.data.split_seed}")
    print()

    manifest_path = processed_dir / "windows_manifest.csv"
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        print("Run script 02_prepare_windows.py first.")
        sys.exit(1)

    manifest = pd.read_csv(manifest_path)
    print(f"Loaded manifest: {len(manifest):,} windows from "
          f"{manifest['source_clip_idx'].nunique():,} source clips")

    # Compute acoustic fingerprints at the source-clip level
    fingerprints_df = compute_source_fingerprints(manifest, sample_rate=cfg.data.sample_rate)

    print("\nRunning cluster-based split assignment...")
    fingerprints_with_splits = generate_splits(
        fingerprints_df,
        train_frac=cfg.data.train_frac,
        val_frac=cfg.data.val_frac,
        n_clusters_drone=cfg.data.n_clusters_drone,
        n_clusters_nondrone=cfg.data.n_clusters_nondrone,
        seed=cfg.data.split_seed,
    )

    # Propagate split labels from source clip → individual windows
    split_map = fingerprints_with_splits.set_index("source_clip_idx")["split"].to_dict()
    manifest["split"] = manifest["source_clip_idx"].map(split_map)

    n_unmapped = manifest["split"].isna().sum()
    if n_unmapped > 0:
        print(f"WARNING: {n_unmapped} windows could not be assigned to a split.")

    # Save individual split manifests
    for split_name in ["train", "val", "test"]:
        split_df = manifest[manifest["split"] == split_name][
            ["clip_id", "file_path", "label", "split", "source_clip_idx"]
        ].copy()
        path = splits_dir / f"{split_name}.csv"
        split_df.to_csv(path, index=False)

    # Also save the full manifest with splits for reference
    manifest.to_csv(splits_dir / "all_windows.csv", index=False)

    # Summary
    print("\n" + "=" * 60)
    print("SPLIT SUMMARY")
    print("=" * 60)
    print(f"  {'Split':8} {'Windows':>10} {'Drone':>10} {'No Drone':>10} {'Balance':>10}")
    print(f"  {'-'*52}")

    for split_name in ["train", "val", "test"]:
        split_df = manifest[manifest["split"] == split_name]
        n = len(split_df)
        nd = int((split_df["label"] == 1).sum())
        nn = int((split_df["label"] == 0).sum())
        ratio = f"{nd/max(nn,1):.2f}:1"
        pct = f"({100*n/len(manifest):.0f}%)"
        print(f"  {split_name:8} {n:>10,}{pct:>6}  {nd:>8,}  {nn:>8,}  {ratio:>10}")

    print(f"\n  Source clips per split:")
    for split_name in ["train", "val", "test"]:
        split_df = manifest[manifest["split"] == split_name]
        n_sources = split_df["source_clip_idx"].nunique()
        print(f"    {split_name}: {n_sources:,} unique source clips")

    print(f"\n  Split manifests saved to: {splits_dir}")
    print(f"\nNEXT STEP: python scripts/04_verify_splits.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate source-aware train/val/test splits.")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config file.",
    )
    main(parser.parse_args())
