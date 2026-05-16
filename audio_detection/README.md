# UAV Audio Detection

A principled audio classification system for detecting UAV (drone) presence in acoustic recordings.

Designed with a robustness-first mindset: the system is built to generalize to realistic deployment conditions, not just to score well on held-out samples from the same dataset.

---

## The Problem This Project Solves

Standard audio classification pipelines trained on the DADS dataset achieve very high accuracy on random held-out splits, but fail badly on real-world recordings. This project identifies and fixes the root causes:

| Problem | Fix Applied |
|---|---|
| Random splits leak recording sessions | Source-aware clustering-based splits |
| Clip duration encodes the label (drone=short, non-drone=long) | All inputs normalized to fixed-length windows |
| Model trained only on clean, isolated clips | Stochastic background mixing during training |
| Single accuracy number hides failure modes | Multi-tier evaluation protocol |

---

## Setup

**Requirements:** Python 3.10+, ~10GB disk space for the dataset.

```bash
# Clone and enter project
cd "Audio Classification model"

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Pipeline (run in order)

```bash
# 1. Audit the dataset — understand what you have before touching model code
python scripts/01_audit_dataset.py

# 2. Download and extract fixed-length windows from all clips
python scripts/02_prepare_windows.py

# 3. Generate source-aware train/val/test splits
python scripts/03_generate_splits.py

# 4. Verify no source leakage across splits
python scripts/04_verify_splits.py

# 5. Train the model
python scripts/05_train.py

# 6. Evaluate across all tiers
python scripts/06_evaluate.py
```

All scripts accept `--config configs/default.yaml` and support `--help`.

---

## Project Structure

```
.
├── configs/            # Hyperparameters and paths (YAML)
├── data/
│   ├── raw/            # Original data — never modified
│   ├── processed/      # Fixed-length windows (.npy files)
│   ├── splits/         # Train/val/test manifests (CSV) — version controlled
│   └── backgrounds/    # Background audio library for augmentation
├── docs/               # Design documents (read these before editing src/)
├── outputs/
│   ├── checkpoints/    # Model checkpoints
│   ├── logs/           # Training logs
│   └── reports/        # Evaluation reports and audit plots
├── scripts/            # Numbered pipeline scripts (01 → 06)
├── src/
│   ├── data/           # Windowing, augmentation, features, splitting, dataset
│   ├── models/         # Model base class and implementations
│   ├── training/       # Trainer, loss functions
│   ├── evaluation/     # Metrics, evaluator, report generation
│   └── utils/          # Config loading, audio I/O
├── tests/              # Unit tests
└── notebooks/          # Exploratory analysis (not part of the pipeline)
```

---

## Key Design Decisions

See [DESIGN.md](DESIGN.md) for the full rationale. Quick summary:

- **Feature:** Log-mel spectrogram (128 mel bins, 10ms frame shift, 16kHz)
- **Input:** 1-second fixed-length windows (removes duration bias)
- **Splits:** Acoustic-feature-based clustering; split at cluster level
- **Augmentation:** Background mixing at SNR ∈ [−5, 20] dB (primary robustness strategy)
- **Loss:** Class-weighted cross-entropy (addresses residual imbalance post-windowing)

---

## Evaluation Tiers

| Tier | What It Measures |
|---|---|
| In-distribution | Sanity check — model learned something |
| Out-of-distribution | Generalization to unseen recording sessions (primary metric) |
| SNR sweep | Robustness at specific noise levels (−10 to +20 dB) |
| Real-world | Acceptance test on actual deployment recordings |

Report metrics at every tier. A single accuracy number is not sufficient.

---

## Dataset

[DADS — Drone Audio Detection Samples](https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples)

- 180,320 samples | 16kHz mono WAV | MIT License
- 163,591 drone clips (avg 0.60s) | 16,729 non-drone clips (avg 7.28s)
- 10 source datasets merged (6 drone, 4 non-drone)
- No explicit source ID column — source grouping is approximated via acoustic clustering
