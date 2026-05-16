#!/usr/bin/env python3
"""
Script 08: Add YouTube drone clips as positive training examples.

Slices downloaded YouTube drone clips into 1-second windows,
saves as .npy files, and appends to train.csv with label=1.

Usage:
    python scripts/08_add_youtube_clips.py
    python scripts/08_add_youtube_clips.py --oversample 10  # repeat each window N times
    python scripts/08_add_youtube_clips.py --dry-run        # preview without writing
    python scripts/08_add_youtube_clips.py --remove         # remove previously added rows
"""

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

TARGET_SR = 16000
WINDOW_SAMPLES = TARGET_SR  # 1 second
OUT_DIR = Path("data/processed/youtube_drone")
TRAIN_CSV = Path("data/splits/train.csv")
SOURCE_TAG = "youtube"

CLIPS = [
    "/tmp/drone_test.wav",   # N3cYbv_9D-g, centroid ~3252Hz
    "/tmp/drone_clip2.wav",  # t4FoCnVLgag, centroid ~2705Hz
    "/tmp/clip3.wav",        # i9zuaGglJkM, centroid ~2947Hz
    "/tmp/clip4.wav",        # oaxX_m-oliY, centroid ~2535Hz
]


def slice_clip(wav_path: str) -> list[np.ndarray]:
    """Load a wav, resample to 16kHz, slice into 1-second non-overlapping windows."""
    waveform, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    n_windows = len(waveform) // WINDOW_SAMPLES
    windows = []
    for i in range(n_windows):
        chunk = waveform[i * WINDOW_SAMPLES : (i + 1) * WINDOW_SAMPLES]
        # Normalize to prevent clipping
        max_val = np.max(np.abs(chunk))
        if max_val > 0:
            chunk = chunk / max_val * 0.95
        windows.append(chunk.astype(np.float32))
    return windows


def main(args: argparse.Namespace) -> None:
    # Change to project root
    project_root = Path(__file__).parent.parent
    out_dir = project_root / OUT_DIR
    train_csv = project_root / TRAIN_CSV

    if not train_csv.exists():
        print(f"ERROR: {train_csv} not found. Run scripts 02-04 first.")
        sys.exit(1)

    df = pd.read_csv(train_csv)

    if args.remove:
        before = len(df)
        df = df[df.get("source_clip_idx", pd.Series(dtype=str)).astype(str) != SOURCE_TAG]
        # Handle case where source_clip_idx column may be named differently
        # Use clip_id prefix as fallback
        if len(df) == before:
            df = df[~df["clip_id"].str.startswith("youtube_")]
        print(f"Removed {before - len(df)} YouTube rows from train.csv.")
        if not args.dry_run:
            df.to_csv(train_csv, index=False)
            print(f"Saved {train_csv}")
        return

    # Remove any previous YouTube rows (idempotent)
    before = len(df)
    df = df[~df["clip_id"].str.startswith("youtube_")]
    if len(df) < before:
        print(f"Removed {before - len(df)} existing YouTube rows (refreshing).")

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    new_rows = []
    total_windows = 0

    for wav_path in CLIPS:
        p = Path(wav_path)
        if not p.exists():
            print(f"WARNING: {wav_path} not found, skipping.")
            continue

        windows = slice_clip(wav_path)
        stem = p.stem  # e.g. "drone_test"
        print(f"  {p.name}: {len(windows)} windows")

        for i, window in enumerate(windows):
            npy_path = out_dir / f"{stem}_{i:04d}.npy"
            rel_path = str((OUT_DIR / f"{stem}_{i:04d}.npy"))

            if not args.dry_run:
                np.save(str(project_root / npy_path), window)

            for rep in range(args.oversample):
                clip_id = f"youtube_{stem}_{i:04d}_r{rep}"
                new_rows.append({
                    "clip_id": clip_id,
                    "file_path": rel_path,
                    "label": 1,
                    "split": "train",
                    "source_clip_idx": SOURCE_TAG,
                })
                total_windows += 1

    print(f"\nTotal new windows: {total_windows}")

    if args.dry_run:
        print("[DRY RUN] No files written.")
        return

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([df, new_df], ignore_index=True)
    combined.to_csv(train_csv, index=False)

    n_drone = int((combined["label"] == 1).sum())
    n_nondrone = int((combined["label"] == 0).sum())
    print(f"\nUpdated train.csv: {len(combined)} rows total")
    print(f"  Drone:     {n_drone}")
    print(f"  Non-drone: {n_nondrone}")
    print(f"\nNEXT STEP: python scripts/05_train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add YouTube drone clips to training data.")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing files.")
    parser.add_argument("--remove", action="store_true", help="Remove previously added YouTube rows.")
    parser.add_argument("--oversample", type=int, default=1,
                        help="Repeat each YouTube window N times in train.csv (default: 1).")
    main(parser.parse_args())
