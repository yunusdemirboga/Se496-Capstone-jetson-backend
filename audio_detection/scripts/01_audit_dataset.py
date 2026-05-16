#!/usr/bin/env python3
"""
Script 01: Dataset Audit

Streams through the DADS dataset from HuggingFace and computes:
  - Class distribution and duration statistics
  - Acoustic fingerprints (spectral centroid, bandwidth, rolloff, RMS)
  - Visualizations: duration histograms, spectral scatter plots, example spectrograms

Run this BEFORE any other script. Understanding the dataset is the
foundation of every subsequent decision.

Usage:
    python scripts/01_audit_dataset.py
    python scripts/01_audit_dataset.py --sample-size 5000 --output-dir outputs/reports/audit
"""

import argparse
import json
import sys
from pathlib import Path

import librosa
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for script use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_clip_stats(waveform: np.ndarray, sr: int = 16000) -> dict:
    """Compute scalar acoustic summary statistics for a single clip."""
    duration = len(waveform) / sr

    waveform = waveform.astype(np.float32)
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val

    centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)[0]
    rms = librosa.feature.rms(y=waveform)[0]

    return {
        "duration_sec": float(duration),
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "spectral_rolloff_mean": float(np.mean(rolloff)),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
    }


def plot_duration_distribution(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (label, name, color) in zip(
        axes,
        [(1, "Drone", "steelblue"), (0, "No Drone", "coral")],
    ):
        subset = df[df["label"] == label]["duration_sec"]
        ax.hist(subset, bins=60, color=color, alpha=0.85, edgecolor="white")
        ax.axvline(
            subset.mean(), color="black", linestyle="--", linewidth=1.5,
            label=f"Mean: {subset.mean():.2f}s",
        )
        ax.axvline(
            subset.median(), color="dimgray", linestyle=":", linewidth=1.5,
            label=f"Median: {subset.median():.2f}s",
        )
        ax.set_title(f"{name} — Duration Distribution\n(n={len(subset):,} sampled)", fontsize=12)
        ax.set_xlabel("Duration (seconds)")
        ax.set_ylabel("Count")
        ax.legend()

    plt.suptitle(
        "Duration Distribution by Class\n"
        "NOTE: Duration is a strong discriminating feature — this is leakage risk.",
        fontsize=12, y=1.04,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_spectral_features(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {0: "coral", 1: "steelblue"}
    labels_map = {0: "No Drone", 1: "Drone"}

    feature_pairs = [
        ("duration_sec", "spectral_centroid_mean", "Duration (s)", "Spectral Centroid (Hz)"),
        ("spectral_centroid_mean", "spectral_bandwidth_mean", "Spectral Centroid (Hz)", "Spectral Bandwidth (Hz)"),
        ("spectral_centroid_mean", "rms_mean", "Spectral Centroid (Hz)", "RMS Energy"),
    ]

    for ax, (x_col, y_col, x_label, y_label) in zip(axes, feature_pairs):
        for label in [0, 1]:
            subset = df[df["label"] == label]
            ax.scatter(
                subset[x_col], subset[y_col],
                c=colors[label], label=labels_map[label],
                alpha=0.35, s=6, rasterized=True,
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(markerscale=3)

    plt.suptitle("Acoustic Feature Space — Separation Between Classes", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_example_spectrograms(examples: dict, output_path: Path) -> None:
    n_cols = min(3, max(len(v) for v in examples.values()))
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8))

    for row, (label, name) in enumerate([(1, "Drone"), (0, "No Drone")]):
        clips = examples.get(label, [])
        for col in range(n_cols):
            ax = axes[row][col] if n_cols > 1 else axes[row]
            if col < len(clips):
                waveform = clips[col].astype(np.float32)
                mel = librosa.feature.melspectrogram(y=waveform, sr=16000, n_mels=128)
                log_mel = librosa.power_to_db(mel, ref=np.max)
                img = librosa.display.specshow(
                    log_mel, sr=16000, hop_length=160,
                    x_axis="time", y_axis="mel", fmax=8000, ax=ax,
                )
                ax.set_title(f"{name} — Example {col + 1}")
                fig.colorbar(img, ax=ax, format="%+2.0f dB")
            else:
                ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("UAV Audio Detection — Dataset Audit")
    print("=" * 60)
    print(f"\nDataset: geronimobasso/drone-audio-detection-samples")
    print(f"Sampling up to {args.sample_size:,} clips per class.")
    print(f"Output:  {output_dir}\n")

    print("Connecting to HuggingFace dataset (streaming)...")
    dataset = load_dataset(
        "geronimobasso/drone-audio-detection-samples",
        split="train",
        streaming=True,
    )

    stats_records = []
    example_waveforms: dict = {0: [], 1: []}
    samples_collected = {0: 0, 1: 0}

    pbar = tqdm(dataset, desc="Streaming clips")
    for sample in pbar:
        label = int(sample["label"])

        if samples_collected[label] >= args.sample_size:
            if all(v >= args.sample_size for v in samples_collected.values()):
                break
            continue

        audio = sample["audio"]
        waveform = np.array(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])

        clip_stats = compute_clip_stats(waveform, sr=sr)
        clip_stats["label"] = label
        stats_records.append(clip_stats)

        if len(example_waveforms[label]) < 3:
            # Keep up to 2 seconds for spectrogram visualization
            max_samples = min(len(waveform), 2 * sr)
            example_waveforms[label].append(waveform[:max_samples])

        samples_collected[label] += 1
        pbar.set_postfix({"drone": samples_collected[1], "non_drone": samples_collected[0]})

    pbar.close()

    df = pd.DataFrame(stats_records)

    # ------------------------------------------------------------------ #
    # Report
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("FULL DATASET CHARACTERISTICS (from dataset card)")
    print("=" * 60)
    print(f"  Drone     (label=1): 163,591 clips | 27.1 hrs | avg 0.60s")
    print(f"  No Drone  (label=0):  16,729 clips | 33.8 hrs | avg 7.28s")
    print(f"  Total:               180,320 clips | 60.9 hrs")
    print(f"  Count imbalance:     ~9.8 : 1  (drone : non-drone)")
    print(f"  Duration bias:       drone avg 0.60s  vs  non-drone avg 7.28s")

    print(f"\n  Known sources: 6 drone + 4 non-drone datasets merged")
    print(f"  Source ID column:    NOT PRESENT (clustering-based split required)")

    print("\n" + "=" * 60)
    print(f"SAMPLED STATISTICS  (n={args.sample_size:,} per class)")
    print("=" * 60)

    for label, name in [(1, "Drone"), (0, "No Drone")]:
        subset = df[df["label"] == label]
        dur = subset["duration_sec"]
        print(f"\n  [{name}]  n={len(subset):,}")
        print(f"    Duration:  mean={dur.mean():.2f}s  std={dur.std():.2f}s  "
              f"min={dur.min():.2f}s  max={dur.max():.2f}s")
        print(f"    Centroid:  {subset['spectral_centroid_mean'].mean():.0f} Hz (mean)")
        print(f"    Bandwidth: {subset['spectral_bandwidth_mean'].mean():.0f} Hz (mean)")
        print(f"    RMS:       {subset['rms_mean'].mean():.4f} (mean)")

    # Duration leakage check
    drone_dur = df[df["label"] == 1]["duration_sec"]
    nondrone_dur = df[df["label"] == 0]["duration_sec"]
    p95_drone = drone_dur.quantile(0.95)
    overlap_frac = (nondrone_dur < p95_drone).mean()

    print(f"\n  Duration Leakage Risk:")
    print(f"    95th percentile drone duration: {p95_drone:.2f}s")
    print(f"    Non-drone clips shorter than this: {overlap_frac*100:.1f}%")
    if overlap_frac < 0.5:
        print(f"    CONCLUSION: Duration separates classes well — HIGH LEAKAGE RISK")
        print(f"    FIX: Fixed-length windowing is mandatory (see script 02).")
    else:
        print(f"    CONCLUSION: Duration overlap is substantial — lower leakage risk.")

    # ------------------------------------------------------------------ #
    # Save stats CSV
    # ------------------------------------------------------------------ #
    stats_path = output_dir / "clip_stats.csv"
    df.to_csv(stats_path, index=False)
    print(f"\nClip statistics saved: {stats_path}")

    # ------------------------------------------------------------------ #
    # Visualizations
    # ------------------------------------------------------------------ #
    print("\nGenerating plots...")

    plot_duration_distribution(df, output_dir / "duration_distribution.png")
    print("  duration_distribution.png")

    plot_spectral_features(df, output_dir / "spectral_features.png")
    print("  spectral_features.png")

    if all(len(v) >= 1 for v in example_waveforms.values()):
        try:
            plot_example_spectrograms(example_waveforms, output_dir / "example_spectrograms.png")
            print("  example_spectrograms.png")
        except Exception as e:
            print(f"  (Skipped spectrogram plot: {e})")

    # ------------------------------------------------------------------ #
    # Summary JSON
    # ------------------------------------------------------------------ #
    summary = {
        "total_clips": 180320,
        "drone_clips": 163591,
        "nondrone_clips": 16729,
        "imbalance_ratio_by_count": round(163591 / 16729, 2),
        "drone_avg_duration_sec": 0.597,
        "nondrone_avg_duration_sec": 7.28,
        "duration_bias_confirmed": True,
        "source_id_available": False,
        "sample_rate_hz": 16000,
        "n_drone_sources": 6,
        "n_nondrone_sources": 4,
    }
    with open(output_dir / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  audit_summary.json")

    print(f"\nAll outputs saved to: {output_dir}")
    print("\n" + "=" * 60)
    print("NEXT STEP: python scripts/02_prepare_windows.py")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Audit the DADS drone audio dataset from HuggingFace."
    )
    parser.add_argument(
        "--sample-size", type=int, default=3000,
        help="Number of clips per class to stream and analyze (default: 3000).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/reports/audit",
        help="Directory for audit outputs.",
    )
    main(parser.parse_args())
