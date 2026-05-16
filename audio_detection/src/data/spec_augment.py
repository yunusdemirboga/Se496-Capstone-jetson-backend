"""
SpecAugment and FreqWarp: spectrogram-level augmentations.

Applied after feature extraction (on the 2D spectrogram array), so it
is feature-type agnostic — works the same on log-mel, CQT, or STFT.

Reference: Park et al., "SpecAugment: A Simple Data Augmentation Method
for Automatic Speech Recognition", Interspeech 2019.

Two masking strategies:
  - Frequency masking: zero out F consecutive mel/frequency bins.
  - Time masking:      zero out T consecutive time frames.

Multiple masks can be applied independently. The mask value is the
global mean of the spectrogram (rather than zero), which avoids
introducing spurious silence artifacts that never appear in real audio.
"""

import random
import numpy as np
from scipy.ndimage import zoom


def apply_freq_mask(
    spec: np.ndarray,
    freq_mask_param: int = 27,
) -> np.ndarray:
    """
    Zero out a random band of frequency bins.

    Args:
        spec:            2D array of shape (freq_bins, time_frames).
        freq_mask_param: Maximum number of consecutive bins to mask (F in paper).

    Returns:
        Masked spectrogram of the same shape.
    """
    freq_bins = spec.shape[0]
    f = random.randint(0, freq_mask_param)
    f0 = random.randint(0, max(freq_bins - f, 0))
    out = spec.copy()
    out[f0:f0 + f, :] = spec.mean()
    return out


def apply_time_mask(
    spec: np.ndarray,
    time_mask_param: int = 30,
) -> np.ndarray:
    """
    Zero out a random band of time frames.

    Args:
        spec:            2D array of shape (freq_bins, time_frames).
        time_mask_param: Maximum number of consecutive frames to mask (T in paper).

    Returns:
        Masked spectrogram of the same shape.
    """
    time_frames = spec.shape[1]
    t = random.randint(0, time_mask_param)
    t0 = random.randint(0, max(time_frames - t, 0))
    out = spec.copy()
    out[:, t0:t0 + t] = spec.mean()
    return out


class SpecAugment:
    """
    Stochastic SpecAugment pipeline for 2D spectrograms.

    Applies independent frequency and time masks. Each mask is sampled
    separately per call, so the same clip looks different each epoch.

    Args:
        freq_mask_param: Max consecutive freq bins to mask per mask.
        time_mask_param: Max consecutive time frames to mask per mask.
        num_freq_masks:  Number of independent frequency masks to apply.
        num_time_masks:  Number of independent time masks to apply.
        prob:            Probability of applying SpecAugment at all.
    """

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 30,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
        prob: float = 0.5,
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.prob = prob

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        """
        Args:
            spec: 2D array (freq_bins, time_frames).

        Returns:
            Augmented spectrogram of the same shape.
        """
        if random.random() >= self.prob:
            return spec

        result = spec.copy()
        for _ in range(self.num_freq_masks):
            result = apply_freq_mask(result, self.freq_mask_param)
        for _ in range(self.num_time_masks):
            result = apply_time_mask(result, self.time_mask_param)
        return result


class FreqWarp:
    """
    Frequency-axis warping augmentation.

    Randomly stretches or compresses the frequency axis of a spectrogram
    by a factor sampled log-uniformly from [1/max_warp, max_warp].

    This simulates drones with different fundamental frequencies without
    requiring new recordings:
        - warp > 1.0 → harmonics shifted upward   (faster/smaller drone)
        - warp < 1.0 → harmonics shifted downward  (slower/larger drone)

    The output is always cropped or zero-padded back to the original
    frequency dimension, preserving the fixed input shape the model expects.

    Args:
        max_warp: Maximum warp factor. 2.0 = up to ±1 octave shift.
        prob:     Probability of applying warp at all.
    """

    def __init__(self, max_warp: float = 2.0, prob: float = 0.4):
        self.max_warp = max_warp
        self.prob = prob

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        """
        Args:
            spec: 2D array (freq_bins, time_frames).

        Returns:
            Warped spectrogram of the same shape.
        """
        if random.random() >= self.prob:
            return spec

        freq_bins, time_frames = spec.shape

        # Log-uniform warp factor in [1/max_warp, max_warp]
        log_warp = random.uniform(-np.log(self.max_warp), np.log(self.max_warp))
        warp = np.exp(log_warp)

        # Zoom only the frequency axis (axis 0); keep time axis (axis 1) fixed
        warped = zoom(spec, zoom=(warp, 1.0), order=1)

        # Crop or pad frequency dimension back to original size
        new_freq = warped.shape[0]
        if new_freq >= freq_bins:
            # Zoomed in (warp > 1): take the centre crop
            start = (new_freq - freq_bins) // 2
            result = warped[start:start + freq_bins, :]
        else:
            # Zoomed out (warp < 1): zero-pad top and bottom
            pad_total = freq_bins - new_freq
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            result = np.pad(warped, ((pad_top, pad_bottom), (0, 0)),
                            mode='constant', constant_values=spec.min())

        return result.astype(np.float32)
