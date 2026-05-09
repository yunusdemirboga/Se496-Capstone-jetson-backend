"""
Fixed-length window extraction.

All audio in this project is converted to fixed-length windows before
any feature extraction or model input. This removes clip duration as an
implicit discriminating feature — the single most important preprocessing
step for avoiding duration leakage.

See DESIGN.md §2.2 for the full rationale.
"""

from typing import Generator

import numpy as np

SAMPLE_RATE = 16000  # Hz — standardized in the DADS dataset


def extract_windows(
    waveform: np.ndarray,
    window_size_sec: float,
    stride_sec: float,
    sample_rate: int = SAMPLE_RATE,
    pad_short: bool = True,
) -> Generator[np.ndarray, None, None]:
    """
    Extract fixed-length windows from a waveform via a sliding window.

    For clips shorter than window_size_sec:
        - If pad_short=True, yields one zero-padded window.
        - If pad_short=False, yields nothing.

    For clips longer than window_size_sec:
        - Yields overlapping windows with the specified stride.
        - If the final partial window contains > 50% content, it is
          zero-padded and yielded.

    Args:
        waveform:        1D numpy array of audio samples (float32).
        window_size_sec: Length of each output window in seconds.
        stride_sec:      Step size between consecutive windows in seconds.
        sample_rate:     Audio sample rate in Hz.
        pad_short:       Whether to pad clips shorter than window_size_sec.

    Yields:
        Fixed-length windows as 1D float32 numpy arrays.
    """
    window_samples = int(window_size_sec * sample_rate)
    stride_samples = int(stride_sec * sample_rate)
    n_samples = len(waveform)

    if n_samples < window_samples:
        if pad_short:
            padded = np.zeros(window_samples, dtype=np.float32)
            padded[:n_samples] = waveform
            yield padded
        return

    start = 0
    while start + window_samples <= n_samples:
        yield waveform[start : start + window_samples].astype(np.float32)
        start += stride_samples

    # Yield final partial window if it contains substantial content
    remaining = n_samples - start
    if remaining > window_samples * 0.5:
        padded = np.zeros(window_samples, dtype=np.float32)
        padded[:remaining] = waveform[start:]
        yield padded


def count_windows(
    duration_sec: float,
    window_size_sec: float,
    stride_sec: float,
) -> int:
    """
    Estimate the number of windows a clip of a given duration will produce.
    Used for progress estimation and balance analysis.
    """
    if duration_sec < window_size_sec:
        return 1  # pad_short=True yields one window
    n_samples = int(duration_sec * SAMPLE_RATE)
    window_samples = int(window_size_sec * SAMPLE_RATE)
    stride_samples = int(stride_sec * SAMPLE_RATE)
    count = (n_samples - window_samples) // stride_samples + 1
    remaining = n_samples - (count - 1) * stride_samples - window_samples
    if remaining > window_samples * 0.5:
        count += 1
    return count
