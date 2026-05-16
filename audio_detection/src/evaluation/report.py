"""
Evaluation report generation.

Produces structured JSON reports and human-readable summaries
from evaluation results. Reports are saved to outputs/reports/.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


def save_report(
    results: Dict,
    model_name: str,
    reports_dir: str = "outputs/reports",
    tag: Optional[str] = None,
) -> str:
    """
    Save evaluation results as a timestamped JSON report.

    Args:
        results:     Dictionary of evaluation results (from Evaluator).
        model_name:  Model identifier.
        reports_dir: Output directory.
        tag:         Optional label appended to filename (e.g., "final").

    Returns:
        Path to the saved report file.
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    filename = f"{model_name}_{timestamp}{suffix}.json"

    report = {
        "model": model_name,
        "timestamp": timestamp,
        "results": results,
    }

    path = reports_dir / filename
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    return str(path)


def print_summary(results: Dict, model_name: str) -> None:
    """Print a human-readable summary of evaluation results to stdout."""
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY — {model_name}")
    print("=" * 60)

    tier_keys = {
        "tier1_in_distribution": "Tier 1 — In-Distribution    (sanity check)",
        "tier2_out_of_distribution": "Tier 2 — Out-of-Distribution (primary metric)",
    }

    for key, label in tier_keys.items():
        if key not in results:
            continue
        m = results[key]
        print(f"\n  {label}")
        print(f"    F1:       {m.get('f1', float('nan')):.3f}")
        print(f"    Precision:{m.get('precision', float('nan')):.3f}")
        print(f"    Recall:   {m.get('recall', float('nan')):.3f}")
        print(f"    PR-AUC:   {m.get('pr_auc', float('nan')):.3f}")
        print(f"    ROC-AUC:  {m.get('roc_auc', float('nan')):.3f}")

    if "tier3_snr_sweep" in results:
        print("\n  Tier 3 — SNR Sweep")
        print(f"  {'SNR (dB)':>10} {'F1':>8} {'Recall':>8} {'Precision':>10}")
        print(f"  {'-'*40}")
        for snr, m in sorted(results["tier3_snr_sweep"].items(), key=lambda x: float(x[0])):
            print(
                f"  {float(snr):>10.0f} "
                f"{m.get('f1', float('nan')):>8.3f} "
                f"{m.get('recall', float('nan')):>8.3f} "
                f"{m.get('precision', float('nan')):>10.3f}"
            )

    print("\n" + "=" * 60)


def plot_snr_curve(
    snr_results: Dict[float, Dict],
    output_path: str,
    model_name: str = "",
) -> None:
    """
    Plot F1, Precision, and Recall vs. SNR level (Tier 3 visualization).

    Args:
        snr_results:  Output of Evaluator.evaluate_snr_sweep().
        output_path:  Where to save the plot.
        model_name:   Used in plot title.
    """
    snr_levels = sorted(float(k) for k in snr_results.keys())
    f1s = [snr_results[snr]["f1"] for snr in snr_levels]
    recalls = [snr_results[snr]["recall"] for snr in snr_levels]
    precisions = [snr_results[snr]["precision"] for snr in snr_levels]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(snr_levels, f1s, "o-", label="F1", linewidth=2)
    ax.plot(snr_levels, recalls, "s--", label="Recall (TPR)", linewidth=1.5, alpha=0.8)
    ax.plot(snr_levels, precisions, "^--", label="Precision", linewidth=1.5, alpha=0.8)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="F1=0.5 threshold")
    ax.axvline(0, color="lightgray", linestyle="-", linewidth=1, label="SNR=0dB")

    ax.set_xlabel("SNR (dB) — higher = drone louder than background")
    ax.set_ylabel("Score")
    ax.set_title(f"Detection Performance vs. SNR{f' — {model_name}' if model_name else ''}")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
