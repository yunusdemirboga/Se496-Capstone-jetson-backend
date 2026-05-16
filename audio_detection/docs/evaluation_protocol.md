# Evaluation Protocol

## Why Standard Evaluation Fails Here

A random train/test split of the DADS dataset produces misleadingly high accuracy because:
- Clips from the same recording session appear in both train and test
- Duration leakage allows trivial classification without acoustic learning

We use a **multi-tier evaluation protocol** that answers different questions at each tier.

---

## Evaluation Tiers

### Tier 1 — In-Distribution (Sanity Check)

**What:** Clips from the same acoustic distribution as training (similar recording conditions, similar drone models), using the held-out test split from `data/splits/test.csv`.

**What it answers:** Did the model learn anything at all?

**Expected performance:** High (F1 > 0.85). If this is low, the training pipeline has a bug.

**Do not optimize for this.** It is a floor, not a ceiling.

---

### Tier 2 — Out-of-Distribution (Primary Metric)

**What:** Clips from recording sessions that are maximally different from training. In practice, these are the test-split clusters that were most acoustically distant from the training clusters.

**What it answers:** Does the model generalize to unseen drones and environments?

**This is the metric we optimize during development.** All architecture and augmentation decisions are evaluated against Tier 2 performance.

**Target:** F1 > 0.70, PR-AUC > 0.75 on OOD test set.

---

### Tier 3 — SNR Sweep (Robustness Benchmark)

**What:** Take clean held-out drone and non-drone clips and programmatically mix them at fixed SNR levels:

```
SNR levels tested: −10, −5, 0, +5, +10, +15, +20 dB
```

At each SNR level, compute the full metric suite and plot a **detection curve**.

**What it answers:** At what noise level does detection break down? What is the system's operational SNR floor?

**Output:** A table and plot of F1 / TPR@FPR5% vs. SNR level. This is the most actionable result for deployment planning.

---

### Tier 4 — Real-World (Acceptance Test)

**What:** Actual recordings of drones in real environments, captured with the target microphone configuration (e.g., laptop microphone recording a drone flying overhead, or recording a drone sound played through a speaker).

**What it answers:** Does the lab performance transfer to deployment conditions?

**How to use:** Manually collect 50–200 real-world samples before any model is declared deployment-ready. This tier is the final gate, not a development tool.

---

## Metrics

All metrics are reported **per tier**.

| Metric | Why |
|---|---|
| Precision | Cost of false alarms |
| Recall | Cost of missed detections |
| F1 | Balanced summary (primary) |
| PR-AUC | Area under precision-recall curve (more informative than ROC under imbalance) |
| ROC-AUC | Threshold-independent discriminability |
| TPR @ FPR=5% | Deployment-relevant operating point |

**Do not report a single accuracy number.** Accuracy is misleading under any class imbalance and hides the trade-off between false positives and false negatives.

---

## Reporting Format

Every evaluation run (`scripts/06_evaluate.py`) produces a structured JSON report in `outputs/reports/`:

```json
{
  "model": "...",
  "timestamp": "...",
  "config": {...},
  "tier1_in_distribution": {
    "precision": 0.91,
    "recall": 0.89,
    "f1": 0.90,
    "pr_auc": 0.94,
    "roc_auc": 0.96,
    "tpr_at_fpr5": 0.88
  },
  "tier2_out_of_distribution": {...},
  "tier3_snr_sweep": {
    "-10": {...},
    "-5": {...},
    "0": {...},
    ...
  }
}
```

---

## What "Better" Means in This Project

In order of priority:
1. Tier 2 F1 ↑
2. Tier 3 F1 at SNR=0dB ↑ (functional at parity noise conditions)
3. Tier 3 F1 at SNR=−5dB ↑ (functional below parity)
4. Tier 1 F1 (informational only)

A model that improves Tier 1 but degrades Tier 2 is a regression, not an improvement.
