"""
Stochastic audio augmentation pipeline.

Augmentation is applied during training only, not pre-computed.
Each training epoch sees different augmented versions of each clip,
providing implicit regularization.

The primary intervention is background mixing, which forces the model
to separate drone acoustic signatures from arbitrary environmental sounds.
This directly targets the real-world deployment failure mode.

Additional waveform-level augmentations:
  - Random gain: simulates different microphone sensitivities / recording levels
  - Room Impulse Response (RIR) simulation: simulates acoustic transmission
    through a room (e.g., phone speaker → air → laptop mic). This is the
    primary intervention for the real-world domain gap.

See DESIGN.md §4 and docs/data_strategy.md §Problem 3 for rationale.
"""

import random
from typing import List, Optional, Tuple

import numpy as np
import librosa
from scipy.signal import fftconvolve


def mix_with_background(
    signal: np.ndarray,
    background: np.ndarray,
    snr_db: float,
    sample_rate: int = 16000,
) -> np.ndarray:
    """
    Mix a signal with a background recording at a specified SNR.

    SNR is defined as 10 * log10(P_signal / P_background).
    Positive SNR: signal louder than background.
    Negative SNR: background louder than signal (challenging condition).

    The background is randomly cropped or tiled to match the signal length.
    The output is peak-normalized to prevent clipping.

    Args:
        signal:     Target signal (drone audio), 1D float32.
        background: Background noise clip, 1D float32.
        snr_db:     Desired SNR in dB.
        sample_rate: Not currently used; reserved for future resampling.

    Returns:
        Mixed signal of same length as `signal`, normalized to [-0.95, 0.95].
    """
    # Match background length to signal
    if len(background) < len(signal):
        repeats = int(np.ceil(len(signal) / len(background)))
        background = np.tile(background, repeats)

    # Random crop of background
    start = random.randint(0, len(background) - len(signal))
    background_segment = background[start : start + len(signal)]

    # Compute RMS energies
    signal_rms = np.sqrt(np.mean(signal**2) + 1e-8)
    background_rms = np.sqrt(np.mean(background_segment**2) + 1e-8)

    # Scale background to achieve the desired SNR
    target_background_rms = signal_rms / (10 ** (snr_db / 20.0))
    scale = target_background_rms / background_rms

    mixed = signal + scale * background_segment

    # Peak-normalize
    max_val = np.max(np.abs(mixed))
    if max_val > 0:
        mixed = mixed / max_val * 0.95

    return mixed.astype(np.float32)


def add_synthetic_noise(
    signal: np.ndarray,
    noise_type: str = "gaussian",
    snr_db: float = 20.0,
) -> np.ndarray:
    """
    Add synthetic noise at a specified SNR.

    Args:
        signal:     Input audio signal, 1D float32.
        noise_type: "gaussian" (white) or "pink" (1/f noise).
        snr_db:     Signal-to-noise ratio in dB.

    Returns:
        Noisy signal of same length as input.
    """
    signal_rms = np.sqrt(np.mean(signal**2) + 1e-8)
    target_noise_rms = signal_rms / (10 ** (snr_db / 20.0))

    if noise_type == "gaussian":
        noise = np.random.randn(len(signal)).astype(np.float32)
    elif noise_type == "pink":
        # Approximate pink noise via cumulative sum of white noise
        white = np.random.randn(len(signal))
        noise = np.cumsum(white).astype(np.float32)
        noise = noise - noise.mean()
    else:
        raise ValueError(f"Unknown noise_type: '{noise_type}'. Use 'gaussian' or 'pink'.")

    noise_rms = np.sqrt(np.mean(noise**2) + 1e-8)
    noise = noise * (target_noise_rms / noise_rms)

    return (signal + noise).astype(np.float32)


def apply_pitch_shift(
    signal: np.ndarray,
    semitones: float,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Shift pitch by a given number of semitones."""
    return librosa.effects.pitch_shift(
        signal.astype(np.float32), sr=sample_rate, n_steps=semitones
    )


def apply_time_stretch(
    signal: np.ndarray,
    rate: float,
    target_length: Optional[int] = None,
) -> np.ndarray:
    """
    Stretch audio by a rate factor, preserving pitch.

    rate > 1.0 speeds up (shorter output).
    rate < 1.0 slows down (longer output).

    If target_length is specified, output is trimmed or zero-padded
    to match exactly. This preserves the fixed-length contract.
    """
    stretched = librosa.effects.time_stretch(signal.astype(np.float32), rate=rate)

    if target_length is not None:
        if len(stretched) >= target_length:
            return stretched[:target_length]
        padded = np.zeros(target_length, dtype=np.float32)
        padded[: len(stretched)] = stretched
        return padded

    return stretched


def apply_random_gain(
    signal: np.ndarray,
    gain_range_db: Tuple[float, float] = (-6.0, 6.0),
) -> np.ndarray:
    """
    Apply a random gain (volume change) to the signal.

    Simulates different microphone sensitivities and recording levels.
    Gain is uniform in dB, which corresponds to a log-uniform scale factor.

    Args:
        signal:        Input audio signal, 1D float32.
        gain_range_db: (min_db, max_db) range for random gain. Negative = quieter.

    Returns:
        Scaled signal. Not peak-normalized — the caller should normalize if needed.
    """
    gain_db = random.uniform(*gain_range_db)
    gain_linear = 10 ** (gain_db / 20.0)
    scaled = signal * gain_linear
    # Clip to prevent hard clipping artifacts
    return np.clip(scaled, -1.0, 1.0).astype(np.float32)


def simulate_rir(
    signal: np.ndarray,
    sample_rate: int = 16000,
    rt60_range_sec: Tuple[float, float] = (0.05, 0.6),
) -> np.ndarray:
    """
    Simulate room acoustics by convolving the signal with a synthetic RIR.

    The impulse response is modelled as exponentially decaying white noise,
    which approximates the diffuse reverberation tail of a real room.

    RT60 is the time for reverberation to decay by 60 dB:
      - 0.05–0.15 s  : small, damped room (closet, studio booth)
      - 0.15–0.35 s  : typical bedroom / office
      - 0.35–0.60 s  : living room, medium hall

    This augmentation directly addresses the phone-speaker → room-air → mic
    domain gap: the model learns that drone harmonics persist even after
    being filtered through an unknown acoustic channel.

    Args:
        signal:          Input audio signal, 1D float32, any length.
        sample_rate:     Sample rate in Hz.
        rt60_range_sec:  (min_rt60, max_rt60) range sampled uniformly.

    Returns:
        Convolved signal of the same length as input, peak-normalized.
    """
    rt60 = random.uniform(*rt60_range_sec)

    # Number of samples for the IR tail
    ir_len = int(rt60 * sample_rate)
    if ir_len < 2:
        return signal

    t = np.arange(ir_len) / sample_rate
    # Decay rate: amplitude drops to 10^(-3) (= -60 dB) at t = rt60
    decay_rate = 3.0 * np.log(10.0) / rt60
    rir = np.random.randn(ir_len).astype(np.float32) * np.exp(-decay_rate * t)
    rir /= np.abs(rir).max() + 1e-8

    # Linear convolution; trim to original length
    convolved = fftconvolve(signal, rir)[:len(signal)]

    max_val = np.abs(convolved).max()
    if max_val > 0:
        convolved = convolved / max_val * 0.95

    return convolved.astype(np.float32)


class AugmentationPipeline:
    """
    Stochastic augmentation pipeline for training.

    Each call applies a random subset of augmentations based on the
    configured probabilities. The pipeline is stateless between calls
    (randomness comes from numpy/random, seeded externally if needed).

    Background mixing is only applied to drone (label=1) clips because:
    - drone + background → still a drone clip (label preserved)
    - non-drone + background → still non-drone (but we don't do this
      to avoid ambiguity if background is accidentally drone-like)

    Usage:
        pipeline = AugmentationPipeline(background_clips=clips)
        augmented = pipeline(waveform, label=1)
    """

    def __init__(
        self,
        background_clips: Optional[List[np.ndarray]] = None,
        snr_range_db: Tuple[float, float] = (-5.0, 20.0),
        background_mix_prob: float = 0.70,
        noise_prob: float = 0.40,
        noise_snr_range_db: Tuple[float, float] = (10.0, 30.0),
        pitch_shift_prob: float = 0.30,
        pitch_shift_range: Tuple[float, float] = (-2.0, 2.0),
        time_stretch_prob: float = 0.20,
        time_stretch_range: Tuple[float, float] = (0.90, 1.10),
        rir_prob: float = 0.40,
        rir_rt60_range_sec: Tuple[float, float] = (0.05, 0.6),
        gain_prob: float = 0.50,
        gain_range_db: Tuple[float, float] = (-6.0, 6.0),
        sample_rate: int = 16000,
    ):
        self.background_clips = background_clips or []
        self.snr_range_db = snr_range_db
        self.background_mix_prob = background_mix_prob
        self.noise_prob = noise_prob
        self.noise_snr_range_db = noise_snr_range_db
        self.pitch_shift_prob = pitch_shift_prob
        self.pitch_shift_range = pitch_shift_range
        self.time_stretch_prob = time_stretch_prob
        self.time_stretch_range = time_stretch_range
        self.rir_prob = rir_prob
        self.rir_rt60_range_sec = rir_rt60_range_sec
        self.gain_prob = gain_prob
        self.gain_range_db = gain_range_db
        self.sample_rate = sample_rate

    def __call__(self, waveform: np.ndarray, label: int) -> np.ndarray:
        """
        Apply stochastic augmentations to a fixed-length waveform.

        Args:
            waveform: 1D float32 audio window.
            label:    0 = no drone, 1 = drone.

        Returns:
            Augmented waveform of the same length as input.
        """
        target_length = len(waveform)
        result = waveform.copy().astype(np.float32)

        # 1. Background mixing — drone clips only
        if (
            label == 1
            and self.background_clips
            and random.random() < self.background_mix_prob
        ):
            background = random.choice(self.background_clips)
            snr = random.uniform(*self.snr_range_db)
            result = mix_with_background(result, background, snr_db=snr)

        # 2. Room Impulse Response — both classes
        # Simulates phone-speaker → room-air → mic acoustic channel.
        if random.random() < self.rir_prob:
            result = simulate_rir(result, sample_rate=self.sample_rate,
                                  rt60_range_sec=self.rir_rt60_range_sec)

        # 3. Random gain — both classes (different mic sensitivities / levels)
        if random.random() < self.gain_prob:
            result = apply_random_gain(result, gain_range_db=self.gain_range_db)

        # 4. Additive synthetic noise — both classes
        if random.random() < self.noise_prob:
            snr = random.uniform(*self.noise_snr_range_db)
            noise_type = random.choice(["gaussian", "pink"])
            result = add_synthetic_noise(result, noise_type=noise_type, snr_db=snr)

        # 5. Pitch shift — both classes (simulates different drone RPM / source pitch)
        if random.random() < self.pitch_shift_prob:
            semitones = random.uniform(*self.pitch_shift_range)
            result = apply_pitch_shift(result, semitones=semitones, sample_rate=self.sample_rate)

        # 6. Time stretch — preserve fixed window length
        if random.random() < self.time_stretch_prob:
            rate = random.uniform(*self.time_stretch_range)
            result = apply_time_stretch(result, rate=rate, target_length=target_length)

        return result
