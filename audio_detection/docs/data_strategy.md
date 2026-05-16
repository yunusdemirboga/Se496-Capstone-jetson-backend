# Data Strategy

## Dataset Overview

**Source:** [DADS — Drone Audio Detection Samples](https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples)

| Property | Value |
|---|---|
| Total samples | 180,320 |
| Drone clips | 163,591 (90.7%) |
| Non-drone clips | 16,729 (9.3%) |
| Drone avg duration | 0.60 seconds |
| Non-drone avg duration | 7.28 seconds |
| Sample rate | 16 kHz mono |
| Format | WAV PCM 16-bit |

**10 source datasets merged:**
- Drone (6): Alemadi 2019, SPCup19 Egonoise, DREGON, DroneNoise Database, AUDROK, Yi et al. 2023
- Non-drone (4): UrbanSound8K, TUT Acoustic Scenes 2017, ESC-50, Zequeira 2021

---

## Problem 1: Source Leakage

### What it is
The DADS dataset has no `source_id` column. Clips from the same recording session (same drone, same environment, same microphone) are scattered across the dataset with no grouping identifier. A random train/test split distributes these acoustically-identical clips across both sets, allowing the model to memorize session-specific fingerprints.

### How we address it
We approximate source groups using acoustic feature clustering:

1. Compute per-clip acoustic fingerprint: `[log(duration), spectral_centroid, spectral_bandwidth, spectral_rolloff, rms_energy]`
2. Apply agglomerative clustering (Ward linkage) within each class
3. Assign entire clusters to train/val/test — never split a cluster

The number of clusters (`n_clusters_drone=10`, `n_clusters_nondrone=6`) is set to approximate the known number of source datasets. Run `scripts/04_verify_splits.py` to confirm no leakage.

---

## Problem 2: Duration Leakage

### What it is
Drone clips average 0.60s; non-drone clips average 7.28s. Any model receiving variable-length inputs (or statistics computed over the full clip) can use duration as a trivial decision rule without learning acoustic content.

### How we address it
All audio is processed through a fixed-length windowing step before any feature extraction:

```
window_size = 1.0 second
stride_train = 0.5 seconds   (50% overlap — more diversity)
stride_eval  = 1.0 second    (non-overlapping — no double counting)
```

- **Short clips (< 1s):** zero-padded to 1 second, yielding 1 window
- **Long clips (≥ 1s):** sliding window extraction

### Effect on class balance

After windowing (approximate counts):
- Drone: ~163,591 clips × ~1 window = ~163,591 windows
- Non-drone: ~16,729 clips × ~13 windows (7.28s / 0.5s stride) = ~217,477 windows
- **Post-windowing ratio: ~0.75:1 (drone:non-drone)** — nearly balanced

This is a natural correction. We do not discard any data.

---

## Problem 3: Low Drone Diversity

### What it is
Many drone clips originate from similar controlled recording conditions. The model risks learning dataset-specific spectral fingerprints rather than generalizable drone acoustics.

### How we address it
Stochastic augmentation during training:

| Augmentation | Probability | Range | Purpose |
|---|---|---|---|
| Background mixing | 70% | SNR −5 to +20 dB | Primary robustness intervention |
| Additive noise | 40% | SNR 10–30 dB | Microphone/environment noise |
| Pitch shift | 30% | ±2 semitones | Different drone models / RPM |
| Time stretch | 20% | 0.9×–1.1× | Temporal variation |

Background mixing is the highest-leverage intervention. It forces the model to separate drone acoustic signatures from arbitrary environmental sounds, which is exactly the skill needed for real-world deployment.

---

## Background Sound Library

For background mixing augmentation, a library of diverse environmental sounds is needed. These must be:
- **Separate from the train/val/test data** (not from DADS)
- Diverse in acoustic content (urban, natural, indoor, outdoor)

**Recommended sources:**
- ESC-50 background classes (wind, rain, sea waves, etc.)
- DEMAND noise database (realistic noise in 18 environments)
- UrbanSound8K (traffic, construction, crowd — if not overlapping with test set)

Download these to `data/backgrounds/`. The `02_prepare_windows.py` script will report if this directory is empty.

---

## Data Flow

```
HuggingFace (raw)
       │
       ▼
scripts/02_prepare_windows.py
       │  Fixed-length windowing (1s)
       │  Amplitude normalization
       │  Save as .npy files
       ▼
data/processed/
       │  drone/00000000.npy ... (one file per window)
       │  non_drone/00000000.npy ...
       ▼
scripts/03_generate_splits.py
       │  Acoustic fingerprint clustering
       │  Cluster-level split assignment
       ▼
data/splits/
       │  train.csv | val.csv | test.csv
       ▼
src/data/dataset.py  (DroneAudioDataset)
       │  Feature extraction (log-mel)
       │  Stochastic augmentation (training only)
       ▼
DataLoader → Model
```
