#!/usr/bin/env python3
"""
Script 07: Generate Synthetic Drone Audio

Generates physically-modelled synthetic drone clips covering the full
frequency range of real-world rotary-wing UAVs:

  - Large consumer drones (DJI-style):  f0 =   80–200 Hz
  - Medium prosumer drones:             f0 =  200–500 Hz
  - Small / racing drones:              f0 =  500–1500 Hz

The DADS training dataset covers mostly the first range. This script
fills in the gap so the model learns the harmonic *pattern* (periodically
spaced tonal peaks) rather than a specific frequency fingerprint.

Acoustic model:
    signal(t) = sum_k [ A_k * (1 + d*sin(2π*f0*t)) * sin(2π*k*f0*t + φ_k) ]
                + noise(t)

    where:
        k       = harmonic index (1 ... N)
        A_k     = 1 / k^alpha  (harmonic amplitude decay)
        d       = AM modulation depth (blade rhythm)
        phi_k   = random phase per harmonic
        noise   = broadband turbulence noise at specified SNR

Output:
    - .npy files in data/processed/synthetic_drone/
    - Appended rows in data/splits/train.csv (label=1, split=train)
    - Val and test splits are NOT modified (real audio only)

Usage:
    python scripts/07_generate_synthetic_drones.py
    python scripts/07_generate_synthetic_drones.py --n-clips 5000
    python scripts/07_generate_synthetic_drones.py --n-clips 2000 --seed 99
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Acoustic model
# ---------------------------------------------------------------------------

def generate_drone_clip(
    sample_rate: int = 16000,
    duration: float = 1.0,
    f0: float = None,
    f0_range: tuple = (80.0, 1500.0),
    n_harmonics_range: tuple = (4, 12),
    harmonic_decay_range: tuple = (0.8, 2.0),
    am_depth_range: tuple = (0.0, 0.3),
    noise_snr_db_range: tuple = (10.0, 35.0),
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Generate one synthetic drone clip.

    Physical model:
        - Tonal component: sum of N harmonics at f0, 2f0, ..., N*f0
        - Harmonic amplitudes: 1/k^alpha (lower harmonics are louder)
        - Amplitude modulation at f0 (blade rhythm makes it pulsate)
        - Broadband turbulence noise added at specified SNR

    f0 is sampled log-uniformly so small drones (high f0) are as
    well-represented as large drones (low f0).

    Args:
        sample_rate:          Output sample rate in Hz.
        duration:             Clip length in seconds.
        f0:                   Fundamental frequency in Hz. If None, sampled
                              log-uniformly from f0_range.
        f0_range:             (min_hz, max_hz) for log-uniform f0 sampling.
        n_harmonics_range:    (min, max) number of harmonics to include.
        harmonic_decay_range: (min, max) for the harmonic amplitude decay
                              exponent alpha.
        am_depth_range:       (min, max) amplitude modulation depth [0, 1].
        noise_snr_db_range:   (min, max) SNR for added broadband noise in dB.
        rng:                  NumPy random Generator (for reproducibility).

    Returns:
        1D float32 array of shape (sample_rate * duration,), peak-normalised.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_samples = int(sample_rate * duration)
    t = np.arange(n_samples) / sample_rate

    # --- Sample parameters ---
    if f0 is None:
        # Log-uniform: equal representation across octaves
        log_min, log_max = np.log(f0_range[0]), np.log(f0_range[1])
        f0 = float(np.exp(rng.uniform(log_min, log_max)))

    n_harmonics = rng.integers(n_harmonics_range[0], n_harmonics_range[1] + 1)
    alpha = rng.uniform(*harmonic_decay_range)
    am_depth = rng.uniform(*am_depth_range)
    noise_snr_db = rng.uniform(*noise_snr_db_range)

    # --- Tonal component ---
    tonal = np.zeros(n_samples, dtype=np.float64)
    for k in range(1, n_harmonics + 1):
        freq = k * f0
        if freq >= sample_rate / 2:
            break  # above Nyquist
        amplitude = 1.0 / (k ** alpha)
        phase = rng.uniform(0, 2 * np.pi)
        harmonic = amplitude * np.sin(2 * np.pi * freq * t + phase)

        # Amplitude modulation: blade rotation creates a pulsing envelope
        am_envelope = 1.0 + am_depth * np.sin(2 * np.pi * f0 * t)
        tonal += am_envelope * harmonic

    # --- Broadband turbulence noise ---
    noise = rng.standard_normal(n_samples)
    # Pink-ish noise via low-pass: more realistic turbulence spectrum
    noise = np.convolve(noise, np.ones(8) / 8, mode='same')

    tonal_rms = np.sqrt(np.mean(tonal ** 2) + 1e-12)
    noise_rms = np.sqrt(np.mean(noise ** 2) + 1e-12)
    target_noise_rms = tonal_rms / (10 ** (noise_snr_db / 20.0))
    noise = noise * (target_noise_rms / noise_rms)

    signal = (tonal + noise).astype(np.float32)

    # Peak-normalise to [-0.95, 0.95]
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.95

    return signal


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_batch(
    n_clips: int,
    output_dir: Path,
    sample_rate: int = 16000,
    seed: int = 42,
) -> list[dict]:
    """
    Generate n_clips synthetic drone clips and save as .npy files.

    Frequency coverage is split evenly across three ranges so no drone
    type is over- or under-represented:
        - 1/3 large consumer drones  (f0:   80–200 Hz)
        - 1/3 medium prosumer drones (f0:  200–500 Hz)
        - 1/3 small / racing drones  (f0:  500–1500 Hz)

    Returns:
        List of manifest row dicts (clip_id, file_path, label, split).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # Equal thirds across the three drone frequency ranges
    ranges = [
        (80.0,   200.0),   # large consumer
        (200.0,  500.0),   # medium prosumer
        (500.0, 1500.0),   # small / racing
    ]
    per_range = n_clips // len(ranges)
    remainder = n_clips % len(ranges)

    clips_per_range = [per_range + (1 if i < remainder else 0)
                       for i in range(len(ranges))]

    rows = []
    clip_index = 0

    print(f"Generating {n_clips} synthetic drone clips...")
    print(f"  Large consumer  (80–200 Hz):   {clips_per_range[0]} clips")
    print(f"  Medium prosumer (200–500 Hz):  {clips_per_range[1]} clips")
    print(f"  Small / racing  (500–1500 Hz): {clips_per_range[2]} clips")
    print()

    for range_idx, (f0_min, f0_max) in enumerate(ranges):
        n = clips_per_range[range_idx]
        for _ in range(n):
            clip_id = f"synthetic_drone_{clip_index:06d}"
            filename = f"{clip_index:06d}.npy"
            file_path = output_dir / filename

            signal = generate_drone_clip(
                sample_rate=sample_rate,
                f0_range=(f0_min, f0_max),
                rng=rng,
            )
            np.save(str(file_path), signal)

            rows.append({
                "clip_id": clip_id,
                "file_path": str(output_dir / filename),
                "label": 1,
                "split": "train",
                "source_clip_idx": -1,
            })

            clip_index += 1
            if clip_index % 500 == 0:
                print(f"  {clip_index}/{n_clips} clips generated...")

    print(f"  {n_clips}/{n_clips} clips generated.")
    return rows


# ---------------------------------------------------------------------------
# Manifest update
# ---------------------------------------------------------------------------

def update_manifest(train_csv: Path, new_rows: list[dict]) -> None:
    """
    Append synthetic clip rows to the training manifest.

    Existing synthetic rows are removed first so re-running the script
    is idempotent (won't keep stacking duplicates).
    """
    df = pd.read_csv(train_csv)

    # Drop any previously generated synthetic rows
    original_count = len(df)
    df = df[~df["clip_id"].str.startswith("synthetic_", na=False)]
    dropped = original_count - len(df)
    if dropped > 0:
        print(f"  Removed {dropped} previously generated synthetic rows.")

    new_df = pd.DataFrame(new_rows)
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(train_csv, index=False)
    print(f"  Train manifest updated: {len(df)} total rows "
          f"({len(df[df.label == 1])} drone, {len(df[df.label == 0])} non-drone)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "processed" / "synthetic_drone"
    train_csv = project_root / "data" / "splits" / "train.csv"

    if not train_csv.exists():
        print(f"ERROR: {train_csv} not found. Run scripts 02-04 first.")
        sys.exit(1)

    print("=" * 60)
    print("UAV Audio Detection — Generate Synthetic Drones")
    print("=" * 60)

    rows = generate_batch(
        n_clips=args.n_clips,
        output_dir=output_dir,
        seed=args.seed,
    )

    print("\nUpdating training manifest...")
    update_manifest(train_csv, rows)

    print(f"\nDone. Synthetic clips saved to: {output_dir}")
    print("NEXT STEP: python scripts/05_train.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic drone audio for training data augmentation."
    )
    parser.add_argument(
        "--n-clips", type=int, default=5000,
        help="Number of synthetic clips to generate (default: 5000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    main(parser.parse_args())
