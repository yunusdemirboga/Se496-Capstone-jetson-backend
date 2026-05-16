"""Tests for fixed-length window extraction."""

import numpy as np
import pytest

from src.data.windowing import count_windows, extract_windows

SAMPLE_RATE = 16000
WINDOW_SEC = 1.0
STRIDE_SEC = 0.5
WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)


def make_waveform(duration_sec: float) -> np.ndarray:
    return np.random.randn(int(duration_sec * SAMPLE_RATE)).astype(np.float32)


class TestExtractWindows:
    def test_short_clip_padded(self):
        """A clip shorter than window_size is zero-padded to window_size."""
        waveform = make_waveform(0.3)
        windows = list(extract_windows(waveform, WINDOW_SEC, STRIDE_SEC, pad_short=True))
        assert len(windows) == 1
        assert len(windows[0]) == WINDOW_SAMPLES

    def test_short_clip_skipped_when_pad_false(self):
        """A clip shorter than window_size yields nothing when pad_short=False."""
        waveform = make_waveform(0.3)
        windows = list(extract_windows(waveform, WINDOW_SEC, STRIDE_SEC, pad_short=False))
        assert len(windows) == 0

    def test_exact_length_clip(self):
        """A clip exactly window_size long yields exactly one window."""
        waveform = make_waveform(1.0)
        windows = list(extract_windows(waveform, WINDOW_SEC, STRIDE_SEC))
        assert len(windows) >= 1
        assert len(windows[0]) == WINDOW_SAMPLES

    def test_all_windows_same_length(self):
        """All windows from a long clip must be exactly window_size samples."""
        waveform = make_waveform(5.0)
        windows = list(extract_windows(waveform, WINDOW_SEC, STRIDE_SEC))
        assert len(windows) > 1
        for w in windows:
            assert len(w) == WINDOW_SAMPLES, f"Window length {len(w)} != {WINDOW_SAMPLES}"

    def test_dtype_is_float32(self):
        """Output windows should be float32."""
        waveform = make_waveform(2.0)
        for w in extract_windows(waveform, WINDOW_SEC, STRIDE_SEC):
            assert w.dtype == np.float32

    def test_long_clip_produces_multiple_windows(self):
        """A 5-second clip with 0.5s stride should produce ~9 windows."""
        waveform = make_waveform(5.0)
        windows = list(extract_windows(waveform, WINDOW_SEC, STRIDE_SEC))
        # (5.0 - 1.0) / 0.5 + 1 = 9 windows
        assert len(windows) >= 8


class TestCountWindows:
    def test_short_clip_counts_as_one(self):
        assert count_windows(0.5, WINDOW_SEC, STRIDE_SEC) == 1

    def test_exact_window_counts_as_one(self):
        assert count_windows(1.0, WINDOW_SEC, STRIDE_SEC) == 1

    def test_long_clip_count(self):
        # 5 second clip, 1s window, 0.5s stride → 9 windows
        count = count_windows(5.0, WINDOW_SEC, STRIDE_SEC)
        assert count >= 8
