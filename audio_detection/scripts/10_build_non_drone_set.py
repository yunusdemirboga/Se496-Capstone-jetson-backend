#!/usr/bin/env python3
"""
Script 10: Build expanded non-drone training set.

Slices three sources into 1-second windows and appends to train.csv
with label=0 (non-drone). No oversampling — each window appears once.

Sources:
  1. data/backgrounds/   — ESC-50 mechanical clips (helicopter, engine,
                           chainsaw, washing machine, vacuum, airplane)
  2. data/raw_non_drone/music/   — downloaded music clips (6 tracks)
  3. data/raw_non_drone/speech/  — downloaded speech clips (5 clips)

Each source is capped at MAX_WINDOWS_PER_FILE windows to prevent any
single clip from dominating the non-drone class.

Usage:
    python scripts/10_build_non_drone_set.py
    python scripts/10_build_non_drone_set.py --dry-run
    python scripts/10_build_non_drone_set.py --remove   # undo additions
"""

import argparse
import sys
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

TARGET_SR       = 16000
WINDOW_SAMPLES  = TARGET_SR          # 1 second
MAX_WINDOWS_PER_FILE = 90            # ~90 seconds per source file
TRAIN_CSV       = Path("data/splits/train.csv")
OUT_DIR         = Path("data/processed/non_drone_expanded")
SOURCE_TAG      = "expanded_non_drone"

# ESC-50 category indices to use from backgrounds/
# These are mechanical/ambient — acoustically adjacent to drones
BACKGROUND_CATEGORIES = {
    35: "washing_machine",
    36: "vacuum_cleaner",
    40: "helicopter",
    41: "chainsaw",
    44: "engine",
    47: "airplane",
}
BACKGROUNDS_DIR = Path("data/backgrounds")
MAX_BACKGROUND_CLIPS = 50  # clips per category (40 exist, take all)


def extract_category(filename: str) -> int:
    """Extract ESC-50 category index from filename like '1-100032-A-0.wav'."""
    try:
        return int(filename.rsplit("-", 1)[-1].replace(".wav", ""))
    except (ValueError, IndexError):
        return -1


_FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def load_and_slice(path: Path, max_windows: int) -> list[np.ndarray]:
    """Load any audio file via ffmpeg, resample to 16kHz, slice into 1-second windows."""
    import subprocess
    max_duration = max_windows  # seconds — only decode what we need
    try:
        result = subprocess.run(
            [_FFMPEG, "-i", str(path),
             "-t", str(max_duration),   # stop after N seconds
             "-ar", str(TARGET_SR),     # resample to 16kHz
             "-ac", "1",               # mono
             "-f", "f32le",            # raw float32 PCM
             "-"],
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"    [warn] ffmpeg failed on {path.name}")
            return []
        waveform = np.frombuffer(result.stdout, dtype=np.float32)
    except Exception as e:
        print(f"    [warn] could not load {path.name}: {e}")
        return []

    n_windows = min(len(waveform) // WINDOW_SAMPLES, max_windows)
    windows = []
    for i in range(n_windows):
        chunk = waveform[i * WINDOW_SAMPLES: (i + 1) * WINDOW_SAMPLES]
        max_val = np.max(np.abs(chunk))
        if max_val > 0:
            chunk = chunk / max_val * 0.95
        windows.append(chunk.astype(np.float32))
    return windows


def process_source(
    label: str,
    files: list[Path],
    out_subdir: Path,
    dry_run: bool,
    max_windows_per_file: int = MAX_WINDOWS_PER_FILE,
) -> list[dict]:
    """Slice a list of audio files into windows and return manifest rows."""
    rows = []
    total_windows = 0

    for path in files:
        windows = load_and_slice(path, max_windows_per_file)
        if not windows:
            continue

        stem = path.stem
        print(f"    {path.name}: {len(windows)} windows")

        for i, window in enumerate(windows):
            rel_path = str(out_subdir / f"{stem}_{i:04d}.npy")
            abs_path = Path(__file__).parent.parent / rel_path

            if not dry_run:
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(abs_path), window)

            rows.append({
                "clip_id":        f"{SOURCE_TAG}_{label}_{stem}_{i:04d}",
                "file_path":      rel_path,
                "label":          0,
                "split":          "train",
                "source_clip_idx": SOURCE_TAG,
            })
            total_windows += 1

    return rows


def main(args: argparse.Namespace) -> None:
    project_root = Path(__file__).parent.parent
    train_csv    = project_root / TRAIN_CSV
    out_dir      = project_root / OUT_DIR

    if not train_csv.exists():
        print(f"ERROR: {train_csv} not found. Run scripts 02-04 first.")
        sys.exit(1)

    df = pd.read_csv(train_csv)

    # ---- Remove mode ----
    if args.remove:
        before = len(df)
        df = df[df.get("source_clip_idx", pd.Series(dtype=str)).astype(str) != SOURCE_TAG]
        removed = before - len(df)
        print(f"Removed {removed} expanded non-drone rows from train.csv.")
        if not args.dry_run:
            df.to_csv(train_csv, index=False)
            print(f"Saved {train_csv}")
        return

    # Remove any previous run (idempotent)
    before = len(df)
    df = df[df.get("source_clip_idx", pd.Series(dtype=str)).astype(str) != SOURCE_TAG]
    if len(df) < before:
        print(f"Removed {before - len(df)} stale expanded rows (refreshing).")

    all_rows = []

    # ------------------------------------------------------------------ #
    # Source 1: ESC-50 mechanical backgrounds
    # ------------------------------------------------------------------ #
    print("\n[1/3] ESC-50 mechanical backgrounds")
    bg_dir = project_root / BACKGROUNDS_DIR
    if not bg_dir.exists():
        print("  WARNING: data/backgrounds/ not found — skipping")
    else:
        for cat_idx, cat_name in BACKGROUND_CATEGORIES.items():
            files = [
                f for f in bg_dir.glob("*.wav")
                if extract_category(f.name) == cat_idx
            ][:MAX_BACKGROUND_CLIPS]
            print(f"  [{cat_idx}] {cat_name}: {len(files)} clips")
            rows = process_source(
                label=cat_name,
                files=files,
                out_subdir=OUT_DIR / "backgrounds",
                dry_run=args.dry_run,
                max_windows_per_file=MAX_WINDOWS_PER_FILE,
            )
            all_rows.extend(rows)

    # ------------------------------------------------------------------ #
    # Source 2: Music clips
    # ------------------------------------------------------------------ #
    print("\n[2/3] Music clips")
    music_dir = project_root / Path("data/raw_non_drone/music")
    if not music_dir.exists() or not list(music_dir.iterdir()):
        print("  WARNING: data/raw_non_drone/music/ is empty — skipping")
    else:
        music_files = list(music_dir.glob("*.*"))
        rows = process_source(
            label="music",
            files=music_files,
            out_subdir=OUT_DIR / "music",
            dry_run=args.dry_run,
            max_windows_per_file=MAX_WINDOWS_PER_FILE,
        )
        all_rows.extend(rows)

    # ------------------------------------------------------------------ #
    # Source 3: Speech clips
    # ------------------------------------------------------------------ #
    print("\n[3/3] Speech clips")
    speech_dir = project_root / Path("data/raw_non_drone/speech")
    if not speech_dir.exists() or not list(speech_dir.iterdir()):
        print("  WARNING: data/raw_non_drone/speech/ is empty — skipping")
    else:
        speech_files = list(speech_dir.glob("*.*"))
        rows = process_source(
            label="speech",
            files=speech_files,
            out_subdir=OUT_DIR / "speech",
            dry_run=args.dry_run,
            max_windows_per_file=MAX_WINDOWS_PER_FILE,
        )
        all_rows.extend(rows)

    # ------------------------------------------------------------------ #
    # Summary and CSV update
    # ------------------------------------------------------------------ #
    print(f"\nTotal new non-drone windows: {len(all_rows)}")

    if args.dry_run:
        print("[DRY RUN] No files written.")
        return

    new_df   = pd.DataFrame(all_rows)
    combined = pd.concat([df, new_df], ignore_index=True)
    combined.to_csv(train_csv, index=False)

    n_drone    = int((combined["label"] == 1).sum())
    n_nondrone = int((combined["label"] == 0).sum())

    print(f"\nUpdated train.csv: {len(combined)} rows total")
    print(f"  Drone:     {n_drone}")
    print(f"  Non-drone: {n_nondrone}  (was {int((df['label']==0).sum())})")
    print(f"\nNEXT STEP: python scripts/05_train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build expanded non-drone training set."
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing files.")
    parser.add_argument("--remove", action="store_true",
                        help="Remove previously added expanded rows.")
    main(parser.parse_args())
