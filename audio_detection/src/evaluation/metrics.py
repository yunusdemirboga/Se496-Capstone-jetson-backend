"""
Evaluation metrics for drone detection.

All metrics are computed per evaluation tier. Never report a single
number — the full metric suite tells a complete story.

See docs/evaluation_protocol.md for the tier definitions and what
each metric answers.
"""

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    fpr_operating_point: float = 0.05,
    threshold: float = 0.5,
) -> Dict:
    """
    Compute the full metric suite for one evaluation tier.

    Args:
        y_true:               Ground truth labels (0 or 1), shape (n,).
        y_pred:               Predicted labels (0 or 1), shape (n,).
        y_prob:               Predicted probability of class 1 (drone), shape (n,).
        fpr_operating_point:  FPR at which to report TPR (deployment operating point).
        threshold:            Classification threshold used to produce y_pred.

    Returns:
        Dictionary of metric_name -> float value.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        # Can fail if only one class present in y_true
        roc_auc = pr_auc = float("nan")

    # TPR at fixed FPR
    fpr, tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
    tpr_at_fpr = float(np.interp(fpr_operating_point, fpr, tpr))

    fpr_key = f"tpr_at_fpr{int(fpr_operating_point * 100)}"

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        fpr_key: tpr_at_fpr,
        "threshold": threshold,
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
    }


def compute_snr_sweep_metrics(
    y_true_by_snr: Dict[float, np.ndarray],
    y_pred_by_snr: Dict[float, np.ndarray],
    y_prob_by_snr: Dict[float, np.ndarray],
    fpr_operating_point: float = 0.05,
) -> Dict[float, Dict]:
    """
    Compute metrics at each SNR level for Tier 3 (robustness benchmark).

    Args:
        y_true_by_snr: {snr_db: ground_truth_array}
        y_pred_by_snr: {snr_db: predictions_array}
        y_prob_by_snr: {snr_db: probabilities_array}
        fpr_operating_point: FPR level for TPR reporting.

    Returns:
        {snr_db: metrics_dict} ordered by ascending SNR.
    """
    return {
        snr: compute_metrics(
            y_true_by_snr[snr],
            y_pred_by_snr[snr],
            y_prob_by_snr[snr],
            fpr_operating_point=fpr_operating_point,
        )
        for snr in sorted(y_true_by_snr.keys())
    }


def find_threshold_at_fpr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_fpr: float = 0.05,
) -> float:
    """
    Find the classification threshold that achieves approximately target_fpr.

    Useful for choosing an operating threshold before deployment.

    Args:
        y_true:     Ground truth labels.
        y_prob:     Predicted probabilities for class 1.
        target_fpr: Desired false positive rate.

    Returns:
        Threshold value in [0, 1].
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob, pos_label=1)
    idx = np.argmin(np.abs(fpr - target_fpr))
    return float(thresholds[idx])
