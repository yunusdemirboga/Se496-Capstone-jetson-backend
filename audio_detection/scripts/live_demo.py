#!/usr/bin/env python3
"""
Live microphone drone detection demo.

Captures audio from your default microphone in 1-second windows and
classifies each window as drone or non-drone in real time.

Usage:
    python scripts/live_demo.py --checkpoint outputs/checkpoints/cnn_baseline/best.pt
    python scripts/live_demo.py --checkpoint outputs/checkpoints/cnn_baseline/best.pt --threshold 0.6
"""

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import sounddevice as sd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.features import extract_log_mel
from src.models.cnn_baseline import CNNBaseline
from src.models.panns_cnn14 import PANNSCNN14
from src.utils.config import load_config

TARGET_SR = 16000


def load_model(checkpoint_path: str, model_type: str):
    if model_type == "panns":
        model = PANNSCNN14(pretrained=False, freeze_backbone=False)
    else:
        model = CNNBaseline()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict(model, waveform: np.ndarray, cfg, model_type: str) -> float:
    """Run inference on a 1-second waveform. Returns drone probability."""
    if model_type == "panns":
        tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
    else:
        features = extract_log_mel(
            waveform,
            sample_rate=cfg.data.sample_rate,
            n_fft=cfg.data.n_fft,
            hop_length=cfg.data.hop_length,
            n_mels=cfg.data.n_mels,
            f_min=cfg.data.f_min,
            f_max=cfg.data.f_max,
            top_db=cfg.data.top_db,
        )
        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=-1)[0]
        return float(probs[1])


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    model = load_model(args.checkpoint, args.model)

    print("=" * 50)
    print("UAV Live Detection Demo")
    print("=" * 50)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Threshold:  {args.threshold}")
    # Detect device native sample rate
    device_info = sd.query_devices(kind="input")
    native_sr = int(device_info["default_samplerate"])
    window_samples = native_sr  # 1 second at native rate

    print(f"Microphone sample rate: {native_sr} Hz (resampling to {TARGET_SR} Hz)")
    print("Press Ctrl+C to stop.\n")

    from collections import deque
    prob_buffer = deque(maxlen=args.smooth_windows)

    try:
        while True:
            # Record 1 second at native sample rate
            audio = sd.rec(
                window_samples,
                samplerate=native_sr,
                channels=1,
                dtype="float32",
                blocking=True,
            )
            waveform = audio[:, 0]

            # Resample to 16kHz if needed
            if native_sr != TARGET_SR:
                waveform = librosa.resample(waveform, orig_sr=native_sr, target_sr=TARGET_SR)

            # Normalize
            max_val = np.max(np.abs(waveform))
            rms = float(np.sqrt(np.mean(waveform ** 2)))
            if max_val > 0:
                waveform = waveform / max_val * 0.95

            raw_prob = predict(model, waveform, cfg, args.model)
            prob_buffer.append(raw_prob)

            # Smoothed probability: average over last N windows
            drone_prob = float(np.mean(prob_buffer))
            is_drone = drone_prob >= args.threshold

            bar_len = 30
            filled = int(drone_prob * bar_len)
            bar = "#" * filled + "-" * (bar_len - filled)

            label = "DRONE     ***" if is_drone else "no drone"
            print(f"[{bar}] {drone_prob:.2f}  ->  {label}  (mic level: {rms:.4f})")

            if args.debug:
                features = extract_log_mel(
                    waveform,
                    sample_rate=cfg.data.sample_rate,
                    n_fft=cfg.data.n_fft,
                    hop_length=cfg.data.hop_length,
                    n_mels=cfg.data.n_mels,
                    f_min=cfg.data.f_min,
                    f_max=cfg.data.f_max,
                    top_db=cfg.data.top_db,
                )
                print(f"  spectrogram shape={features.shape} min={features.min():.1f} max={features.max():.1f} mean={features.mean():.1f}")

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live drone audio detection demo.")
    parser.add_argument(
        "--checkpoint", type=str, default="outputs/checkpoints/cnn_baseline/best.pt",
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.35,
        help="Drone probability threshold (default: 0.35).",
    )
    parser.add_argument(
        "--smooth-windows", type=int, default=3,
        help="Number of windows to average for temporal smoothing (default: 3).",
    )
    parser.add_argument(
        "--model", type=str, default="cnn_baseline", choices=["cnn_baseline", "panns"],
        help="Model type: cnn_baseline or panns (default: cnn_baseline).",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print spectrogram stats for each window.",
    )
    main(parser.parse_args())
