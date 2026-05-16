"""
Audio I/O utilities.

Thin wrappers for consistent audio loading across the project.
All audio is expected to be 16kHz mono float32 in the range [-1, 1].
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import librosa
import soundfile as sf


def load_audio(
    path: str,
    target_sr: int = 16000,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and resample to target_sr.

    Returns:
        (waveform, sample_rate) where waveform is float32.
    """
    waveform, sr = librosa.load(path, sr=target_sr, mono=mono)
    return waveform.astype(np.float32), target_sr


def normalize_amplitude(waveform: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Peak-normalize audio to target amplitude."""
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        return (waveform / max_val * target_peak).astype(np.float32)
    return waveform.astype(np.float32)


def save_audio(path: str, waveform: np.ndarray, sample_rate: int = 16000) -> None:
    """Save a waveform as a 16-bit PCM WAV file."""
    sf.write(path, waveform, sample_rate, subtype="PCM_16")


def load_background_clips(directory: str, sample_rate: int = 16000) -> List[np.ndarray]:
    """
    Load all audio files from a directory for use as augmentation backgrounds.

    Recursively searches for .wav, .mp3, and .flac files.

    Args:
        directory: Path to background sounds library (data/backgrounds/).
        sample_rate: Target sample rate for all clips.

    Returns:
        List of normalized waveform arrays. Empty list if directory is empty.
    """
    clips = []
    directory = Path(directory)

    if not directory.exists():
        return clips

    for pattern in ("**/*.wav", "**/*.mp3", "**/*.flac"):
        for path in sorted(directory.glob(pattern)):
            try:
                waveform, _ = load_audio(str(path), target_sr=sample_rate)
                clips.append(normalize_amplitude(waveform))
            except Exception as e:
                print(f"Warning: Could not load background clip {path}: {e}")

    return clips
