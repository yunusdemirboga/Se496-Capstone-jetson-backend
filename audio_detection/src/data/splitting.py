"""
Source-aware dataset splitting.

The DADS dataset merges 10 source datasets with no explicit source_id
column. A random train/test split places clips from the same recording
session in both sets, causing the model to memorize session fingerprints.

We approximate source groups via acoustic feature clustering:
  - Compute a fingerprint for each clip (duration, spectral centroid,
    spectral bandwidth, spectral rolloff, RMS energy)
  - Cluster within each class separately (drone / non-drone)
  - Assign entire clusters to train / val / test

Splitting is done once and the result is committed to data/splits/.
See DESIGN.md §5 for full rationale.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


def compute_fingerprints(df: pd.DataFrame) -> np.ndarray:
    """
    Compute normalized acoustic fingerprints for clustering.

    Expected input columns:
        duration_sec, spectral_centroid_mean, spectral_bandwidth_mean,
        spectral_rolloff_mean, rms_mean

    Log(duration) is used instead of raw duration to compress the
    large range of duration values in the dataset (0.02s – 228s).

    Returns:
        Normalized feature matrix of shape (n_clips, 5).
    """
    df = df.copy()
    df["log_duration"] = np.log1p(df["duration_sec"])

    cols = [
        "log_duration",
        "spectral_centroid_mean",
        "spectral_bandwidth_mean",
        "spectral_rolloff_mean",
        "rms_mean",
    ]
    X = df[cols].values
    return StandardScaler().fit_transform(X)


def cluster_clips(fingerprints: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Cluster clips by acoustic fingerprint using agglomerative clustering.

    Ward linkage minimizes within-cluster variance, producing compact,
    similarly-sized clusters that approximate recording sessions.

    Args:
        fingerprints: Normalized feature matrix (n_clips, n_features).
        n_clusters:   Number of clusters. Set to approximate the number
                      of known source datasets for each class.

    Returns:
        Array of cluster IDs of shape (n_clips,).
    """
    return AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward",
    ).fit_predict(fingerprints)


def assign_clusters_to_splits(
    cluster_ids: np.ndarray,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> np.ndarray:
    """
    Assign entire clusters to train / val / test splits.

    Clusters are assigned as atomic units — never split across sets.
    Assignment is greedy: clusters are shuffled and assigned to splits
    until each split's target fraction is reached.

    Args:
        cluster_ids: Per-clip cluster assignments.
        train_frac:  Target fraction for training set.
        val_frac:    Target fraction for validation set.
        seed:        Random seed for reproducibility.

    Returns:
        Array of split labels ("train", "val", "test") of shape (n_clips,).
    """
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(cluster_ids)
    total = len(cluster_ids)

    cluster_sizes = {c: int(np.sum(cluster_ids == c)) for c in unique_clusters}

    shuffled = list(rng.permutation(unique_clusters))
    splits = np.empty(total, dtype=object)

    # Reserve one cluster each for val and test so every split is guaranteed
    # to be non-empty regardless of how the greedy loop fills train.
    reserved_val = reserved_test = None
    if len(shuffled) >= 3:
        reserved_test = shuffled.pop()
        reserved_val = shuffled.pop()

    assigned_train = assigned_val = 0

    for cluster_id in shuffled:
        size = cluster_sizes[cluster_id]
        mask = cluster_ids == cluster_id

        current_train_frac = assigned_train / total
        current_val_frac = assigned_val / total

        if current_train_frac < train_frac:
            splits[mask] = "train"
            assigned_train += size
        elif current_val_frac < val_frac:
            splits[mask] = "val"
            assigned_val += size
        else:
            splits[mask] = "test"

    if reserved_val is not None:
        splits[cluster_ids == reserved_val] = "val"
    if reserved_test is not None:
        splits[cluster_ids == reserved_test] = "test"

    return splits


def generate_splits(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    n_clusters_drone: int = 10,
    n_clusters_nondrone: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate source-aware splits for the full dataset.

    Processes drone (label=1) and non-drone (label=0) clips separately,
    then merges results. The two classes live in different regions of
    acoustic feature space, so separate clustering is more meaningful.

    Args:
        df:                DataFrame with columns: clip_id, label, duration_sec,
                           spectral_centroid_mean, spectral_bandwidth_mean,
                           spectral_rolloff_mean, rms_mean.
        train_frac:        Target training fraction.
        val_frac:          Target validation fraction.
        n_clusters_drone:  Source clusters for drone class (≈ known source count).
        n_clusters_nondrone: Source clusters for non-drone class.
        seed:              Random seed for determinism.

    Returns:
        Input DataFrame with an added "split" column.
    """
    result = df.copy()
    result["split"] = None

    for label, n_clusters in [(1, n_clusters_drone), (0, n_clusters_nondrone)]:
        mask = df["label"] == label
        subset = df[mask].copy()

        if len(subset) == 0:
            continue

        # Ensure we don't request more clusters than samples
        n_clusters = min(n_clusters, len(subset))

        fingerprints = compute_fingerprints(subset)
        cluster_ids = cluster_clips(fingerprints, n_clusters=n_clusters)
        split_labels = assign_clusters_to_splits(
            cluster_ids, train_frac=train_frac, val_frac=val_frac, seed=seed
        )
        result.loc[mask, "split"] = split_labels

    return result


def verify_no_source_leakage(
    manifest: pd.DataFrame,
    source_col: str = "source_clip_idx",
) -> Tuple[bool, dict]:
    """
    Verify that no source clip contributes windows to multiple splits.

    A source clip should never have windows in both train and test.
    If leakage is detected, the split generation step should be rerun.

    Args:
        manifest:   DataFrame with 'source_clip_idx' and 'split' columns.
        source_col: Column identifying the original source clip.

    Returns:
        (is_clean, report) where is_clean=True means no leakage detected,
        and report contains per-source split assignment details.
    """
    source_splits = manifest.groupby(source_col)["split"].nunique()
    leaking_sources = source_splits[source_splits > 1]

    report = {
        "total_sources": int(source_splits.shape[0]),
        "leaking_sources": int(len(leaking_sources)),
        "leaking_source_ids": leaking_sources.index.tolist()[:20],  # show first 20
    }

    return len(leaking_sources) == 0, report
