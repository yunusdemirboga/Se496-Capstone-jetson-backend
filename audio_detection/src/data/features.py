"""
Audio feature extraction.

Drone acoustics — rotary-wing UAVs — produce tonal, harmonic signals
with fundamental frequencies typically in the 80–400 Hz range. The
log-mel spectrogram is the default representation because it:

  - Preserves tonal and harmonic structure across the relevant range
  - Uses a logarithmic frequency axis (harmonics appear equally spaced)
  - Is computationally efficient
  - Has strong precedent in audio classification literature

See docs/feature_representation.md for the full justification, and
for why MFCCs were rejected.
"""

from typing import Literal

import numpy as np
import librosa

FeatureType = Literal["log_mel", "cqt", "stft"]


def extract_log_mel(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 160,
    n_mels: int = 128,
    f_min: float = 20.0,
    f_max: float = 8000.0,
    top_db: float = 80.0,
) -> np.ndarray:
    """
    Compute a log-mel spectrogram.

    For a 1-second window at 16kHz, returns shape (128, 101):
    128 mel bins × 101 time frames (10ms each).

    Values are in dB relative to the maximum, clipped at -top_db.

    Args:
        waveform:    1D float32 audio array.
        sample_rate: Audio sample rate in Hz.
        n_fft:       FFT window size in samples (~64ms at 16kHz).
        hop_length:  Frame shift in samples (160 = 10ms at 16kHz).
        n_mels:      Number of mel filter banks.
        f_min:       Lowest frequency in Hz.
        f_max:       Highest frequency in Hz.
        top_db:      Dynamic range floor in dB.

    Returns:
        2D array of shape (n_mels, time_frames), values in dB.
    """
    mel = librosa.feature.melspectrogram(
        y=waveform.astype(np.float32),
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max,
    )
    return librosa.power_to_db(mel, ref=np.max, top_db=top_db)


def extract_cqt(
    waveform: np.ndarray,
    sample_rate: int = 16000,
    hop_length: int = 160,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    f_min: float = 32.7,  # C1 — covers drone fundamentals from ~33Hz up
) -> np.ndarray:
    """
    Compute a Constant-Q Transform spectrogram.

    Better logarithmic frequency resolution than mel, particularly suited
    for harmonic content. More expensive to compute than log-mel.

    Returns:
        2D array of shape (n_bins, time_frames), values in dB.
    """
    cqt = librosa.cqt(
        y=waveform.astype(np.float32),
        sr=sample_rate,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin=f_min,
    )
    return librosa.amplitude_to_db(np.abs(cqt), ref=np.max)


def extract_features(
    waveform: np.ndarray,
    feature_type: FeatureType = "log_mel",
    sample_rate: int = 16000,
    **kwargs,
) -> np.ndarray:
    """
    Unified feature extraction interface.

    Args:
        waveform:     1D float32 audio array (fixed-length window).
        feature_type: Which representation to compute.
        sample_rate:  Audio sample rate in Hz.
        **kwargs:     Passed to the specific extractor (n_mels, n_fft, etc.).

    Returns:
        2D feature array of shape (freq_bins, time_frames).

    Raises:
        ValueError: If feature_type is not recognized.
    """
    if feature_type == "log_mel":
        return extract_log_mel(waveform, sample_rate=sample_rate, **kwargs)

    if feature_type == "cqt":
        return extract_cqt(waveform, sample_rate=sample_rate, **kwargs)

    if feature_type == "stft":
        stft = librosa.stft(waveform.astype(np.float32), **kwargs)
        return librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    raise ValueError(
        f"Unknown feature_type: '{feature_type}'. "
        "Choose from: 'log_mel', 'cqt', 'stft'."
    )
