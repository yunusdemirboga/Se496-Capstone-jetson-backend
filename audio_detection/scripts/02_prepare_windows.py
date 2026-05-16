#!/usr/bin/env python3
"""
Script 02: Prepare Fixed-Length Windows

Streams the DADS dataset from HuggingFace and converts clips into
fixed-length 1-second windows.

Uses Audio(decode=False) to receive raw bytes from HuggingFace parquet
files, then decodes with soundfile — no FFmpeg or torchcodec required.

Exact duplicate detection:
  Each clip's waveform is MD5-hashed before windowing. Clips that are
  byte-identical to a previously seen clip are skipped. This prevents
  inflated accuracy from duplicate clips appearing across splits.

Near-duplicate handling:
  Near-duplicates (acoustically similar but not identical) are handled
  by the cluster-based splitting in script 03 — similar clips cluster
  together and stay in the same split.

Disk usage (with --max-windows-per-class 5000):
  ~10k windows x 16,000 samples x 4 bytes = ~640 MB as .npy files.

Usage:
    python scripts/02_prepare_windows.py --max-windows-per-class 5000
    python scripts/02_prepare_windows.py  # full dataset, no cap
"""

import argparse
import hashlib
import io
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.windowing import extract_windows
from src.utils.config import load_config


def hash_waveform(waveform: np.ndarray) -> str:
    """MD5 hash of raw waveform bytes — used for exact duplicate detection."""
    return hashlib.md5(waveform.tobytes()).hexdigest()


def decode_audio_bytes(audio_field: dict, target_sr: int = 16000) -> np.ndarray:
    """
    Decode raw audio bytes from a HuggingFace Audio(decode=False) field.

    The field contains either:
      - 'bytes': raw audio file bytes (WAV/FLAC/MP3)
      - 'path':  local file path (fallback)
    """
    raw_bytes = audio_field.get("bytes")
    if raw_bytes:
        waveform, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
    else:
        waveform, sr = sf.read(audio_field["path"], dtype="float32", always_2d=False)

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)

    return waveform.astype(np.float32)


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)

    processed_dir = Path(cfg.data.processed_dir)
    drone_dir = processed_dir / "drone"
    nondrone_dir = processed_dir / "non_drone"

    for d in [drone_dir, nondrone_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UAV Audio Detection — Prepare Windows")
    print("=" * 60)
    print(f"Window size:   {cfg.data.window_size_sec}s")
    print(f"Stride:        {cfg.data.window_stride_sec}s")
    print(f"Sample rate:   {cfg.data.sample_rate} Hz")
    print(f"Output:        {processed_dir}")
    print(f"Dedup:         enabled (MD5 hash per clip)")
    if args.max_windows_per_class:
        print(f"Cap:           {args.max_windows_per_class:,} windows per class (streaming)")
    print()

    # Shuffle before streaming so drone and non-drone clips are interleaved.
    # Without this, the dataset is ordered non-drone first (~16k clips) then
    # drone, meaning we'd scan through thousands of skipped clips to fill both
    # class caps. buffer_size controls how many clips are held in memory for
    # shuffling — larger = better mixing but more RAM.
    dataset = load_dataset(
        "geronimobasso/drone-audio-detection-samples",
        split="train",
        streaming=True,
    ).shuffle(seed=42, buffer_size=2000).cast_column("audio", Audio(decode=False))

    manifest_records = []
    window_idx = {0: 0, 1: 0}
    clips_seen = {0: 0, 1: 0}
    duplicates_skipped = {0: 0, 1: 0}
    seen_hashes = set()
    source_clip_idx = 0

    pbar = tqdm(dataset, desc="Windowing clips")
    for sample in pbar:
        label = int(sample["label"])
        cap = args.max_windows_per_class

        if cap and window_idx[label] >= cap:
            if all(window_idx[c] >= cap for c in [0, 1]):
                break
            continue

        try:
            waveform = decode_audio_bytes(sample["audio"], target_sr=cfg.data.sample_rate)
        except Exception:
            continue

        # Exact duplicate detection
        clip_hash = hash_waveform(waveform)
        if clip_hash in seen_hashes:
            duplicates_skipped[label] += 1
            continue
        seen_hashes.add(clip_hash)

        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val * 0.95

        subdir = drone_dir if label == 1 else nondrone_dir

        for window in extract_windows(
            waveform,
            window_size_sec=cfg.data.window_size_sec,
            stride_sec=cfg.data.window_stride_sec,
            sample_rate=cfg.data.sample_rate,
            pad_short=cfg.data.pad_short_clips,
        ):
            if cap and window_idx[label] >= cap:
                break

            win_id = window_idx[label]
            filename = f"{win_id:08d}.npy"
            file_path = subdir / filename
            np.save(file_path, window)

            manifest_records.append({
                "clip_id": f"{'drone' if label == 1 else 'nondrone'}_{win_id:08d}",
                "file_path": str(file_path),
                "label": label,
                "source_clip_idx": source_clip_idx,
            })
            window_idx[label] += 1

        clips_seen[label] += 1
        source_clip_idx += 1
        pbar.set_postfix({
            "drone": window_idx[1],
            "nondrone": window_idx[0],
            "dups": sum(duplicates_skipped.values()),
        })

    pbar.close()

    manifest = pd.DataFrame(manifest_records)
    manifest_path = processed_dir / "windows_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    drone_n = window_idx[1]
    nondrone_n = window_idx[0]
    total_dups = sum(duplicates_skipped.values())

    print("\n" + "=" * 60)
    print("WINDOW EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"  Source clips used: drone={clips_seen[1]:,}  non-drone={clips_seen[0]:,}")
    print(f"  Duplicates skipped: drone={duplicates_skipped[1]:,}  non-drone={duplicates_skipped[0]:,}  total={total_dups:,}")
    print(f"  Drone windows:     {drone_n:>10,}")
    print(f"  Non-drone windows: {nondrone_n:>10,}")
    print(f"  Total windows:     {drone_n + nondrone_n:>10,}")
    print(f"  Balance ratio:     {drone_n/max(nondrone_n,1):.2f} : 1  (drone : non-drone)")
    print(f"\n  Manifest saved:    {manifest_path}")
    print(f"\nNEXT STEP: python scripts/03_generate_splits.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract fixed-length windows from DADS dataset."
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--max-windows-per-class", type=int, default=None,
        help=(
            "Stop collecting once each class has this many windows. "
            "Uses HuggingFace streaming — only downloads what is needed. "
            "Recommended: 5000 (~640 MB). Omit for the full dataset (~24 GB)."
        ),
    )
    main(parser.parse_args())
