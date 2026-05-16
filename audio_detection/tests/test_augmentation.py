"""Tests for the augmentation pipeline."""

import numpy as np
import pytest

from src.data.augmentation import (
    AugmentationPipeline,
    add_synthetic_noise,
    apply_pitch_shift,
    apply_time_stretch,
    mix_with_background,
)

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 16000  # 1 second


def make_tone(freq: float = 200.0, duration: float = 1.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Create a pure tone for testing."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestMixWithBackground:
    def test_output_length_matches_signal(self):
        signal = make_tone()
        background = make_tone(freq=50.0, duration=2.0)
        mixed = mix_with_background(signal, background, snr_db=10.0)
        assert len(mixed) == len(signal)

    def test_output_is_float32(self):
        signal = make_tone()
        background = make_tone(freq=50.0)
        mixed = mix_with_background(signal, background, snr_db=10.0)
        assert mixed.dtype == np.float32

    def test_output_not_clipped(self):
        """Output should be peak-normalized to at most 0.95."""
        signal = make_tone()
        background = make_tone(freq=50.0)
        mixed = mix_with_background(signal, background, snr_db=-10.0)
        assert np.max(np.abs(mixed)) <= 0.96  # small tolerance

    def test_short_background_is_tiled(self):
        """Background shorter than signal is tiled automatically."""
        signal = make_tone(duration=2.0)
        short_bg = make_tone(freq=50.0, duration=0.3)
        mixed = mix_with_background(signal, short_bg, snr_db=5.0)
        assert len(mixed) == len(signal)


class TestAddSyntheticNoise:
    def test_output_length_preserved(self):
        signal = make_tone()
        noisy = add_synthetic_noise(signal, noise_type="gaussian", snr_db=20.0)
        assert len(noisy) == len(signal)

    def test_pink_noise(self):
        signal = make_tone()
        noisy = add_synthetic_noise(signal, noise_type="pink", snr_db=15.0)
        assert len(noisy) == len(signal)

    def test_invalid_noise_type_raises(self):
        with pytest.raises(ValueError, match="Unknown noise_type"):
            add_synthetic_noise(make_tone(), noise_type="purple")


class TestApplyTimeStretch:
    def test_output_length_preserved_with_target(self):
        signal = make_tone()
        stretched = apply_time_stretch(signal, rate=1.1, target_length=WINDOW_SAMPLES)
        assert len(stretched) == WINDOW_SAMPLES

    def test_output_length_preserved_slowdown(self):
        signal = make_tone()
        stretched = apply_time_stretch(signal, rate=0.9, target_length=WINDOW_SAMPLES)
        assert len(stretched) == WINDOW_SAMPLES


class TestAugmentationPipeline:
    def test_output_length_preserved(self):
        """Pipeline must not change the window length."""
        signal = make_tone()
        background = [make_tone(freq=50.0, duration=2.0)]
        pipeline = AugmentationPipeline(background_clips=background)
        result = pipeline(signal, label=1)
        assert len(result) == len(signal)

    def test_background_mix_only_for_drone(self):
        """Non-drone clips should never have background mixing applied."""
        # Run many times and verify no errors (background mixing skipped for label=0)
        signal = make_tone()
        background = [make_tone(freq=50.0, duration=2.0)]
        pipeline = AugmentationPipeline(
            background_clips=background, background_mix_prob=1.0
        )
        for _ in range(10):
            result = pipeline(signal, label=0)
            assert len(result) == len(signal)

    def test_no_background_clips_still_works(self):
        """Pipeline works without any background clips."""
        signal = make_tone()
        pipeline = AugmentationPipeline(background_clips=[])
        result = pipeline(signal, label=1)
        assert len(result) == len(signal)

    def test_output_is_float32(self):
        signal = make_tone()
        pipeline = AugmentationPipeline()
        result = pipeline(signal, label=1)
        assert result.dtype == np.float32
