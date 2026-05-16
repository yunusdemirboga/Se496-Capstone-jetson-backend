"""
audio_client.py — UAV audio detection pipeline for Jetson Orin NX.

Captures audio from the ReSpeaker mic array in 1-second windows, runs
CNN inference, and optionally sends detection events to the backend via
WebSocket when a state change occurs (UAV appeared / UAV lost).

Usage:
    # Verify setup without mic (simulated audio):
    python3 audio_client.py --simulate

    # Live mic, no backend yet (prints detections to terminal):
    python3 audio_client.py --device 0

    # Full pipeline once backend is ready:
    python3 audio_client.py --device 0 \
        --backend ws://192.168.x.x:8000/ws/detections \
        --station-id <base_station_uuid>

    # List available audio devices:
    python3 audio_client.py --list-devices

IMPORTANT: Run from the audio_detection/ directory so config paths resolve:
    cd /home/anis/yolo_inference/audio_detection
    python3 audio_client.py --simulate
"""

import argparse
import asyncio
import json
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

# ---- numpy compat patch — must be before any other import ----
# Restores aliases removed in numpy 1.24 (required by system TensorRT binding).
# Matches the same patch in run_inference.py.
import numpy as np
if not hasattr(np, 'bool'):    np.bool    = np.bool_
if not hasattr(np, 'int'):     np.int     = np.int_
if not hasattr(np, 'float'):   np.float   = np.float_
if not hasattr(np, 'complex'): np.complex = np.complex_
if not hasattr(np, 'object'):  np.object  = np.object_
if not hasattr(np, 'str'):     np.str     = np.str_

import torch

# Ensure src/ is importable when running from audio_detection/
sys.path.insert(0, str(Path(__file__).parent))

from src.data.features import extract_log_mel
from src.models.cnn_baseline import CNNBaseline
from src.utils.config import load_config

# ---- Constants ----
TARGET_SR         = 16000          # model expects 16 kHz mono
CHECKPOINT_PATH   = "outputs/checkpoints/cnn_baseline/best.pt"
CONFIG_PATH       = "configs/default.yaml"
DEBOUNCE_COUNT    = 3              # consecutive windows needed to flip state
SMOOTH_WINDOWS    = 3              # rolling average window for prob smoothing


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str) -> CNNBaseline:
    path = Path(checkpoint_path)
    if not path.exists():
        sys.exit(
            f"ERROR: Checkpoint not found at {path.resolve()}\n"
            "Place best.pt at outputs/checkpoints/cnn_baseline/best.pt"
        )
    model = CNNBaseline()
    ckpt = torch.load(str(path), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(model: CNNBaseline, waveform: np.ndarray, cfg) -> float:
    """Return drone probability (0.0–1.0) for a 1-second waveform."""
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
        return float(torch.softmax(logits, dim=-1)[0][1])


# ---------------------------------------------------------------------------
# Audio capture
# ---------------------------------------------------------------------------

def get_live_chunk(device_index, native_sr: int) -> np.ndarray:
    """Record exactly 1 second at native_sr, return as 16 kHz mono float32."""
    import sounddevice as sd
    audio = sd.rec(
        native_sr,
        samplerate=native_sr,
        channels=1,
        dtype="float32",
        device=device_index,
        blocking=True,
    )
    waveform = audio[:, 0]

    # Resample to 16 kHz if the mic runs at a different native rate
    if native_sr != TARGET_SR:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=native_sr, target_sr=TARGET_SR)

    return waveform


def get_simulated_chunk() -> np.ndarray:
    """Return silent audio — model should output a low drone probability."""
    return np.zeros(TARGET_SR, dtype=np.float32)


def normalize(waveform: np.ndarray) -> np.ndarray:
    """Peak-normalize to 0.95. Matches live_demo.py behaviour."""
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val * 0.95
    return waveform


# ---------------------------------------------------------------------------
# WebSocket sender (optional — only used when --backend is provided)
# ---------------------------------------------------------------------------

async def send_event(ws, station_id: str, drone_detected: bool, confidence: float):
    event_label = "UAV_APPEARED" if drone_detected else "UAV_LOST"
    payload = {
        "base_station_id": station_id,
        "drone_detected": drone_detected,
        "acoustic_confidence": round(confidence, 4),
        "yolo_confidence": None,
        "detected_at": datetime.now(timezone.utc).isoformat(),
        "description": f"[{event_label}] Acoustic detection event. Drone probability: {confidence:.1%}",
    }
    await ws.send(json.dumps(payload))
    try:
        ack = await asyncio.wait_for(ws.recv(), timeout=2.0)
        print(f"  [WS] backend ack: {ack}")
    except asyncio.TimeoutError:
        print("  [WS] no ack from backend (timeout)")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run(args):
    cfg   = load_config(CONFIG_PATH)
    model = load_model(args.checkpoint)

    # Resolve audio device and its native sample rate
    native_sr = TARGET_SR
    if not args.simulate:
        import sounddevice as sd
        device_info = sd.query_devices(
            args.device if args.device is not None else sd.default.device[0],
            kind="input"
        )
        native_sr = int(device_info["default_samplerate"])
        print(f"Mic: {device_info['name']}  native={native_sr} Hz → resampling to {TARGET_SR} Hz")
    else:
        print("Running in SIMULATE mode (no microphone required).")

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Threshold  : {args.threshold}")
    print(f"Debounce   : {DEBOUNCE_COUNT} consecutive windows")
    print(f"Smoothing  : {SMOOTH_WINDOWS}-window rolling average")
    if args.backend:
        print(f"Backend WS : {args.backend}")
    else:
        print("Backend WS : not configured (terminal output only)")
    print("Press Ctrl+C to stop.\n")

    # State machine
    detection_state = "CLEAR"   # "CLEAR" | "DETECTED"
    consecutive     = 0
    prob_buffer     = deque(maxlen=SMOOTH_WINDOWS)

    # Open WebSocket if backend is configured
    ws = None
    ws_context = None
    if args.backend:
        if not args.station_id:
            sys.exit("ERROR: --station-id is required when --backend is set.")
        import websockets
        ws_context = websockets.connect(args.backend)
        ws = await ws_context.__aenter__()
        print("WebSocket connected.\n")

    try:
        while True:
            t0 = time.time()

            # Capture
            if args.simulate:
                waveform = get_simulated_chunk()
            else:
                waveform = get_live_chunk(args.device, native_sr)

            waveform = normalize(waveform)

            # Infer
            raw_prob = predict(model, waveform, cfg)
            prob_buffer.append(raw_prob)
            drone_prob = float(np.mean(prob_buffer))
            is_uav     = drone_prob >= args.threshold

            # Progress bar
            bar_len = 30
            filled  = int(drone_prob * bar_len)
            bar     = "#" * filled + "-" * (bar_len - filled)
            label   = "UAV DETECTED ***" if is_uav else "clear"
            elapsed = time.time() - t0
            print(f"[{bar}] {drone_prob:.2f}  {label}  state={detection_state}  ({elapsed:.2f}s)")

            # Debounce state machine
            if is_uav and detection_state == "CLEAR":
                consecutive += 1
                if consecutive >= DEBOUNCE_COUNT:
                    detection_state = "DETECTED"
                    consecutive     = 0
                    print(f"  >>> EVENT: UAV_APPEARED  (conf={drone_prob:.3f})")
                    if ws:
                        await send_event(ws, args.station_id, True, drone_prob)
            elif not is_uav and detection_state == "DETECTED":
                consecutive += 1
                if consecutive >= DEBOUNCE_COUNT:
                    detection_state = "CLEAR"
                    consecutive     = 0
                    print(f"  >>> EVENT: UAV_LOST  (conf={drone_prob:.3f})")
                    if ws:
                        await send_event(ws, args.station_id, False, drone_prob)
            else:
                consecutive = 0

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if ws_context and ws:
            await ws_context.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="UAV audio detection pipeline for Jetson Orin NX."
    )
    parser.add_argument(
        "--checkpoint", default=CHECKPOINT_PATH,
        help=f"Path to model checkpoint (default: {CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--config", default=CONFIG_PATH,
        help=f"Path to config YAML (default: {CONFIG_PATH})",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Drone probability threshold 0.0-1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--device", type=int, default=None,
        help="sounddevice input device index (use --list-devices to find it)",
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Run with simulated silent audio — no microphone required",
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="Print available audio input devices and exit",
    )
    parser.add_argument(
        "--backend", type=str, default=None,
        help="WebSocket URL of backend, e.g. ws://192.168.1.x:8000/ws/detections",
    )
    parser.add_argument(
        "--station-id", type=str, default=None,
        help="Base station UUID (required when --backend is set)",
    )
    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        sys.exit(0)

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
