"""Tests for source-aware split generation."""

import numpy as np
import pandas as pd
import pytest

from src.data.splitting import (
    assign_clusters_to_splits,
    cluster_clips,
    compute_fingerprints,
    generate_splits,
    verify_no_source_leakage,
)


def make_fingerprint_df(n_drone: int = 100, n_nondrone: int = 50, seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic fingerprint DataFrame for testing."""
    rng = np.random.default_rng(seed)
    records = []
    for i in range(n_drone):
        records.append({
            "clip_id": f"drone_{i:04d}",
            "label": 1,
            "duration_sec": rng.uniform(0.5, 1.0),
            "spectral_centroid_mean": rng.uniform(800, 2000),
            "spectral_bandwidth_mean": rng.uniform(500, 1500),
            "spectral_rolloff_mean": rng.uniform(1000, 4000),
            "rms_mean": rng.uniform(0.01, 0.1),
        })
    for i in range(n_nondrone):
        records.append({
            "clip_id": f"nondrone_{i:04d}",
            "label": 0,
            "duration_sec": rng.uniform(3.0, 10.0),
            "spectral_centroid_mean": rng.uniform(1500, 4000),
            "spectral_bandwidth_mean": rng.uniform(1000, 3000),
            "spectral_rolloff_mean": rng.uniform(2000, 8000),
            "rms_mean": rng.uniform(0.01, 0.15),
        })
    return pd.DataFrame(records)


class TestComputeFingerprints:
    def test_shape(self):
        df = make_fingerprint_df()
        fp = compute_fingerprints(df)
        assert fp.shape == (len(df), 5)

    def test_normalized(self):
        """Fingerprints should be StandardScaler normalized (mean ≈ 0)."""
        df = make_fingerprint_df(n_drone=200, n_nondrone=100)
        fp = compute_fingerprints(df)
        assert abs(fp.mean()) < 0.5


class TestClusterClips:
    def test_n_unique_clusters(self):
        df = make_fingerprint_df()
        fp = compute_fingerprints(df)
        cluster_ids = cluster_clips(fp, n_clusters=5)
        assert len(np.unique(cluster_ids)) == 5

    def test_output_length(self):
        df = make_fingerprint_df()
        fp = compute_fingerprints(df)
        cluster_ids = cluster_clips(fp, n_clusters=4)
        assert len(cluster_ids) == len(df)


class TestAssignClustersToSplits:
    def test_all_assigned(self):
        cluster_ids = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        splits = assign_clusters_to_splits(cluster_ids)
        assert not any(s is None for s in splits)

    def test_valid_split_labels(self):
        cluster_ids = np.repeat(np.arange(10), 10)
        splits = assign_clusters_to_splits(cluster_ids)
        assert set(splits).issubset({"train", "val", "test"})

    def test_deterministic_with_seed(self):
        cluster_ids = np.repeat(np.arange(8), 20)
        s1 = assign_clusters_to_splits(cluster_ids, seed=42)
        s2 = assign_clusters_to_splits(cluster_ids, seed=42)
        assert list(s1) == list(s2)

    def test_fractions_approximate(self):
        """Train fraction should be within 15% of target."""
        cluster_ids = np.repeat(np.arange(20), 50)
        splits = assign_clusters_to_splits(cluster_ids, train_frac=0.70, val_frac=0.15)
        train_frac = np.mean(splits == "train")
        assert 0.55 <= train_frac <= 0.85


class TestGenerateSplits:
    def test_all_rows_assigned(self):
        df = make_fingerprint_df()
        result = generate_splits(df, n_clusters_drone=5, n_clusters_nondrone=3)
        assert result["split"].notna().all()

    def test_valid_split_values(self):
        df = make_fingerprint_df()
        result = generate_splits(df)
        assert set(result["split"]).issubset({"train", "val", "test"})


class TestVerifyNoSourceLeakage:
    def test_clean_manifest_passes(self):
        """Each source clip in only one split → clean."""
        manifest = pd.DataFrame({
            "source_clip_idx": [0, 0, 1, 1, 2, 2, 3, 3],
            "split": ["train", "train", "train", "train", "val", "val", "test", "test"],
        })
        is_clean, report = verify_no_source_leakage(manifest)
        assert is_clean
        assert report["leaking_sources"] == 0

    def test_leaking_manifest_fails(self):
        """Source clip 0 appears in both train and test → not clean."""
        manifest = pd.DataFrame({
            "source_clip_idx": [0, 0, 1, 1],
            "split": ["train", "test", "train", "train"],
        })
        is_clean, report = verify_no_source_leakage(manifest)
        assert not is_clean
        assert report["leaking_sources"] == 1
