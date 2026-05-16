# U-SCAR Detection Pipeline — Jetson Orin NX

Multi-modal UAV detection system combining a YOLO11m vision model (TensorRT) and
a CNN audio classifier. Detections are streamed to a FastAPI backend via WebSocket
and live video is streamed to the dashboard via WebRTC.

## System

| Component | Version |
|---|---|
| Device | Jetson Orin NX |
| JetPack | 5.1.3 |
| CUDA | 11.4 |
| cuDNN | 8.6.0 |
| TensorRT | 8.5.2.2 |
| Python | 3.8.10 |
| OS | Ubuntu 20.04 (aarch64) |

---

## Models

| Model | File | Purpose |
|---|---|---|
| YOLO11m TRT | `best.engine` (41 MB) | Frame-by-frame visual drone detection |
| CNN audio classifier | `audio_detection/outputs/checkpoints/cnn_baseline/best.pt` | 1-second window acoustic detection |

Both models are hardware-specific and **not in git** — see setup steps below.

---

## Quick Start

```bash
# Standard run (vision + audio, headless, no Gemini)
PYTHONUNBUFFERED=1 python3 -u pipeline.py --no-show --no-gpt

# Vision only (skip audio)
PYTHONUNBUFFERED=1 python3 -u pipeline.py --no-show --no-gpt --no-audio

# Test audio without a real microphone
PYTHONUNBUFFERED=1 python3 -u pipeline.py --no-show --no-gpt --simulate-audio

# Dry run (no backend, prints events to terminal)
PYTHONUNBUFFERED=1 python3 -u pipeline.py --no-show --dry-run
```

> **Always use `PYTHONUNBUFFERED=1 python3 -u`** when running in the background
> so logs are flushed immediately.

---

## Environment Variables (`.env`)

```
BACKEND_WS_URL=wss://se496-capstone-dashboard-backend.onrender.com/ws/detections
BASE_STATION_ID=<uuid from dashboard>
SUPABASE_URL=https://unjuiavhmurpcgcqnsby.supabase.co
SUPABASE_KEY=<service role key>
SUPABASE_BUCKET=detections
GEMINI_API_KEY=<optional — needed unless --no-gpt>
SIGNED_URL_EXPIRY_SECONDS=3600
```

---

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--no-show` | off | Headless mode — required on Jetson (no display connected) |
| `--no-gpt` | off | Skip Gemini AI verification |
| `--no-audio` | off | Disable audio detection, run vision only |
| `--simulate-audio` | off | Use silent audio — no microphone required |
| `--audio-device N` | system default | sounddevice input device index |
| `--audio-threshold F` | 0.5 | Audio drone probability threshold (0.0–1.0) |
| `--list-audio-devices` | — | Print audio input devices and exit |
| `--dry-run` | off | Print events to terminal only, no backend |
| `--camera N` | 0 | Camera device index |
| `--conf F` | 0.5 | YOLO confidence threshold |
| `--imgsz N` | 640 | YOLO inference image size |
| `--backend URL` | `BACKEND_WS_URL` | Backend WebSocket URL |
| `--station-id UUID` | `BASE_STATION_ID` | Base station UUID |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      pipeline.py                        │
│                                                         │
│  Main thread (YOLO inference loop)                      │
│  ├── Reads camera frames at 30 fps                      │
│  ├── Runs YOLO11m TRT inference on each frame           │
│  ├── State machine: CLEAR → DETECTED → CLEAR            │
│  │   confirm=3 frames, miss=10 frames, heartbeat=30s    │
│  └── Pushes events to EventWorker queue                 │
│                                                         │
│  AudioWorker thread (daemon)                            │
│  ├── Captures 1-second audio windows from mic           │
│  ├── Runs CNN audio classifier                          │
│  ├── Smooths probability over 3-window rolling average  │
│  └── Updates shared _audio_confidence (thread-safe)     │
│                                                         │
│  EventWorker thread (daemon, asyncio loop)              │
│  ├── Uploads detection frame to Supabase Storage        │
│  ├── Calls Gemini 2.0 Flash for visual verification     │
│  ├── Reads latest acoustic_confidence from AudioWorker  │
│  └── Sends JSON payload via WebSocket to backend        │
│                                                         │
│  WebRTCSignaling thread (daemon, asyncio loop)          │
│  └── Streams annotated video frames to dashboard        │
└─────────────────────────────────────────────────────────┘
```

---

## Event Types

| Event | Trigger | Payload |
|---|---|---|
| `drone_appeared` | 3 consecutive positive YOLO frames (CLEAR → DETECTED) | image, yolo_conf, acoustic_conf, Gemini description |
| `drone_ongoing` | Every 30s while DETECTED (heartbeat) | image, yolo_conf, acoustic_conf |
| `drone_gone` | suppressed (no imageless entries in dashboard) | — |

All events include:
- `base_station_id` — UUID of this Jetson unit
- `drone_detected` — bool (false if Gemini overrides)
- `yolo_confidence` — float
- `acoustic_confidence` — float from CNN audio model
- `image_url` — Supabase storage path (backend converts to public URL)
- `description` — Gemini report or event label
- `detected_at` — ISO 8601 UTC timestamp

---

## Audio Detection

The audio model is a three-block CNN trained on log-mel spectrograms (128 mel bins,
10 ms frame shift, 16 kHz). It classifies 1-second windows as drone / no-drone.

Checkpoint: `audio_detection/outputs/checkpoints/cnn_baseline/best.pt`
Config:     `audio_detection/configs/default.yaml`

The AudioWorker runs in a daemon thread, independently of the YOLO loop.
It updates `acoustic_confidence` every ~1 second. If the checkpoint is missing
or the microphone cannot be opened, the thread logs a warning and exits — the
rest of the pipeline continues with `acoustic_confidence = 0.0`.

To list available audio input devices:
```bash
python3 pipeline.py --list-audio-devices
```

---

## Vision Detection — YOLO11m TRT

- Input: 640×640, FP16 TensorRT engine
- Classes: `0: Not UAV`, `1: UAV` (both treated as positive; see `get_uav_class_ids`)
- Camera opens at 1280×720 @ 30 fps
- Engine load: ~20–30 s on first run

---

## Supabase Image Storage

Frames are uploaded as JPEG (quality 85) to the `detections` bucket.

```
Public URL format:
https://<project>.supabase.co/storage/v1/object/public/detections/detections/<timestamp>_<uid>.jpg
```

The backend generates the public URL from the `file_path` field in the payload.

---

## WebRTC Live Stream

The pipeline connects to `/ws/webrtc/producer` on the backend and streams the
annotated camera feed (bounding boxes + status overlay) to any connected dashboard viewer.

---

## First-Time Setup

```bash
# 1. Install system dependencies
sudo apt-get install -y libopenblas-dev libjpeg-dev zlib1g-dev libpython3-dev \
    libavcodec-dev libavformat-dev libswscale-dev cmake ninja-build

# 2. Run the environment setup script (installs PyTorch NVIDIA wheel, builds torchvision)
bash setup_yolo_jetson.sh

# 3. Install audio dependencies
pip3 install sounddevice librosa

# 4. Download best.pt and place it in yolo_inference/
# 5. Build TensorRT engine (~15 min)
python3 export_to_trt.py

# 6. Copy .env and fill in your values
cp .env.example .env   # edit SUPABASE_KEY, BASE_STATION_ID, BACKEND_WS_URL, etc.

# 7. Run
PYTHONUNBUFFERED=1 python3 -u pipeline.py --no-show --no-gpt
```

---

## Files

| File | Purpose |
|---|---|
| `pipeline.py` | **Main entry point** — runs YOLO + audio + WebRTC + WebSocket |
| `export_to_trt.py` | Converts `best.pt` → `best.engine` (TensorRT FP16) |
| `setup_yolo_jetson.sh` | One-time environment setup |
| `best.engine` | TensorRT engine — hardware-specific, not in git |
| `audio_detection/audio_client.py` | Standalone audio pipeline (for testing audio alone) |
| `audio_detection/src/` | CNN model, feature extraction, config utilities |
| `audio_detection/configs/default.yaml` | Audio model hyperparameters |
| `audio_detection/outputs/checkpoints/cnn_baseline/best.pt` | Audio model weights |
| `backend/` | FastAPI backend (deployed on Render) |
| `.env` | Local environment variables (not in git) |

---

## Verified Package Stack

| Package | Version |
|---|---|
| torch | 2.1.0a0+41361538.nv23.06 |
| torchvision | 0.16.1 |
| numpy | 1.23.5 |
| opencv | 4.5.4 |
| tensorrt | 8.5.2.2 |
| ultralytics | 8.4.21 |
| sounddevice | 0.4.x |
| librosa | 0.10.x |
| websockets | 11.x |
| aiortc | 1.x |
