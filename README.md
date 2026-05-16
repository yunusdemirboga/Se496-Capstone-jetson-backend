# U-SCAR Detection Pipeline — Jetson Orin NX

Multi-modal UAV detection system combining a YOLO11m vision model (TensorRT FP16) and a
CNN audio classifier. Both models run simultaneously in a single pipeline — detections
are streamed to a FastAPI backend via WebSocket and live annotated video is streamed to
the dashboard via WebRTC.

---

## System

| Component | Version |
|---|---|
| Device | NVIDIA Jetson Orin NX 8 GB |
| JetPack | 5.1.3 |
| CUDA | 11.4 |
| cuDNN | 8.6.0 |
| TensorRT | 8.5.2.2 |
| Python | 3.8.10 |
| OS | Ubuntu 20.04 (aarch64) |

---

## Quick Start

```bash
# Full pipeline — YOLO vision + CNN audio + WebRTC + backend (headless, no Gemini)
PYTHONUNBUFFERED=1 python3 -u pipeline.py --no-show --no-gpt --audio-device 0

# Vision only (no microphone)
PYTHONUNBUFFERED=1 python3 -u pipeline.py --no-show --no-gpt --no-audio

# With display connected
PYTHONUNBUFFERED=1 python3 -u pipeline.py --no-gpt --audio-device 0

# Dry run — no backend, prints events to terminal
PYTHONUNBUFFERED=1 python3 -u pipeline.py --no-show --dry-run
```

> Always use `PYTHONUNBUFFERED=1 python3 -u` when running in the background so logs flush immediately.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        pipeline.py                          │
│                                                             │
│  Main thread — YOLO inference loop                          │
│  ├── Camera frames @ 30 fps (1280×720)                      │
│  ├── YOLO11m TRT inference (640×640 FP16, GPU)              │
│  ├── State machine: CLEAR ↔ DETECTED                        │
│  │     confirm = 3 consecutive positive frames              │
│  │     miss    = 10 consecutive negative frames             │
│  │     heartbeat every 30 s while DETECTED                  │
│  └── Pushes events → EventWorker queue                      │
│                                                             │
│  AudioWorker thread (daemon)                                │
│  ├── ReSpeaker mic @ 16 kHz, 1-second windows               │
│  ├── Log-mel spectrogram (128 bins, 10 ms frames)           │
│  ├── CNNBaseline inference (101K params, CPU)               │
│  ├── 3-window rolling average for probability smoothing     │
│  └── Updates shared acoustic_confidence every ~1 s          │
│                                                             │
│  EventWorker thread (daemon, asyncio loop)                  │
│  ├── Uploads JPEG frame → Supabase Storage                  │
│  ├── Calls Gemini 2.0 Flash for visual verification         │
│  ├── Reads latest acoustic_confidence from AudioWorker      │
│  └── Sends JSON payload → backend /ws/detections            │
│                                                             │
│  WebRTCSignaling thread (daemon, asyncio loop)              │
│  ├── Persistent WS to /ws/webrtc/producer                   │
│  └── Streams annotated frames → dashboard browser           │
└─────────────────────────────────────────────────────────────┘
```

---

## Model 1 — YOLO11m Vision (TensorRT FP16)

### Overview

YOLO11m is the medium variant of Ultralytics YOLO11 — a real-time single-stage object
detector. It is converted to a TensorRT FP16 engine for GPU-accelerated inference on
the Jetson Orin NX.

### Architecture

| Property | Value |
|---|---|
| Base architecture | YOLO11m (medium) |
| Framework | Ultralytics 8.4.21 |
| Export format | TensorRT FP16 engine |
| Input resolution | 640 × 640 pixels |
| Input channels | 3 (BGR) |
| Output | Bounding boxes + class scores + confidence |
| Classes | `0: Not UAV`, `1: UAV` |
| Engine size | ~41 MB |
| Engine load time | ~20–30 s (first run) |

### Inference Pipeline

```
Camera frame (1280×720 BGR)
    │
    ▼  resize + normalize
YOLO11m TRT (640×640 FP16 on GPU)
    │
    ▼  NMS + confidence filter (default: conf ≥ 0.5)
Bounding boxes with class + confidence
    │
    ▼  filter to UAV class only
State machine (CLEAR / DETECTED)
    │
    ▼  on state change or heartbeat
Event queue → EventWorker
```

### State Machine

| Parameter | Value | Meaning |
|---|---|---|
| `CONFIRM_FRAMES` | 3 | Consecutive positive frames to transition CLEAR → DETECTED |
| `MISS_FRAMES` | 10 | Consecutive negative frames to transition DETECTED → CLEAR |
| `HEARTBEAT_INTERVAL` | 30 s | Interval for `drone_ongoing` events while DETECTED |
| `JPEG_QUALITY` | 85 | Compression quality for uploaded detection frames |

### Class Handling

Both class IDs are treated as positive UAV detections (the model was trained with
`0: Not UAV` as a background class, but any detection above the confidence threshold
indicates drone presence). The `get_uav_class_ids()` function matches class names
containing `uav`, `drone`, `aerial`, or `quadcopter` — falling back to all classes
if none match.

### Camera

| Property | Value |
|---|---|
| Interface | USB / CSI |
| Requested resolution | 1280 × 720 |
| Frame rate | 30 fps |
| Colour space | BGR (OpenCV default) |

> The Jetson GStreamer driver may report a different actual resolution. YOLO rescales
> internally to 640×640 regardless of camera output resolution.

---

## Model 2 — CNNBaseline Audio Classifier

### Overview

A lightweight convolutional neural network that classifies 1-second audio windows as
**drone** or **no-drone** using log-mel spectrograms. Designed for real-time inference
on CPU (no GPU required for audio).

### Dataset — DADS

| Property | Value |
|---|---|
| Dataset | [DADS — Drone Audio Detection Samples](https://huggingface.co/datasets/geronimobasso/drone-audio-detection-samples) |
| Total clips | 180,320 |
| Sample rate | 16 kHz mono WAV |
| License | MIT |
| Drone clips | 163,591 (avg duration 0.60 s) |
| Non-drone clips | 16,729 (avg duration 7.28 s) |
| Source datasets | 10 merged datasets (6 drone, 4 non-drone) |

### Data Preparation

Raw clips are converted to fixed-length 1-second windows before training. This removes
clip duration as an implicit discriminating feature (drone clips are shorter; non-drone
clips are longer — a model seeing variable-length inputs can exploit this trivially).

| Parameter | Value |
|---|---|
| Window size | 1.0 s |
| Window stride (training) | 0.5 s (50% overlap, more diversity) |
| Window stride (eval) | 1.0 s (non-overlapping, no double-counting) |
| Short clip handling | Zero-padded to 1 s |
| Post-windowing balance | ~163K drone / ~216K non-drone windows (~1:1.4) |

Class imbalance is further addressed by class-weighted cross-entropy loss during training.

### Splits — Source-Aware Clustering

Random 80/20 splits leak recording sessions across train/test, producing falsely high
accuracy. Splits are generated at the **acoustic cluster level**:

1. Compute acoustic fingerprint per clip: log(duration), spectral centroid, bandwidth,
   rolloff, RMS energy
2. Cluster within each class separately using agglomerative clustering (Ward linkage)
3. Assign entire clusters to train / val / test — no clip from the same session appears
   in two splits

| Split | Fraction |
|---|---|
| Train | 70% |
| Val | 15% |
| Test | 15% |

### Feature Extraction — Log-Mel Spectrogram

| Parameter | Value | Rationale |
|---|---|---|
| Feature type | Log-mel spectrogram | Preserves tonal/harmonic structure of rotor acoustics |
| Sample rate | 16,000 Hz | DADS standard; covers drone fundamentals (80–400 Hz) |
| FFT size (`n_fft`) | 1,024 samples (~64 ms) | Frequency resolution vs time resolution tradeoff |
| Hop length | 160 samples (10 ms) | Standard for audio classification |
| Mel bins (`n_mels`) | 128 | Sufficient resolution across 20–8000 Hz |
| Frequency range | 20 Hz – 8,000 Hz | Below: below audible drone range. Above: Nyquist for 16 kHz |
| Dynamic range (`top_db`) | 80 dB | Floor to avoid log(0) and compress noise floor |
| Output shape | (128, 101) | 128 mel bins × 101 time frames per 1-second window |

**Why not MFCC:** MFCCs discard pitch information via cepstral liftering — pitch is a
primary discriminator for drone detection. MFCCs are the wrong prior here.

### Architecture — CNNBaseline

```
Input: (batch, 1, 128, 101)   — 1-channel log-mel spectrogram

Conv Block 1:  Conv2d(1→32,   3×3, pad=1) → BatchNorm → ReLU → MaxPool2d(2×2)
               Output: (batch, 32, 64, 50)

Conv Block 2:  Conv2d(32→64,  3×3, pad=1) → BatchNorm → ReLU → MaxPool2d(2×2)
               Output: (batch, 64, 32, 25)

Conv Block 3:  Conv2d(64→128, 3×3, pad=1) → BatchNorm → ReLU → MaxPool2d(2×2)
               Output: (batch, 128, 16, 12)

Global Average Pool:  AdaptiveAvgPool2d(1)
               Output: (batch, 128)

Classifier:    Linear(128→64) → ReLU → Dropout(0.4) → Linear(64→2)
               Output: (batch, 2)   — logits for [no-drone, drone]
```

| Property | Value |
|---|---|
| Total parameters | 101,506 (~101K) |
| Dropout | 0.4 (classifier head only) |
| Output | Softmax probability for drone class |
| Inference device | CPU |

### Training

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 64 |
| Loss | Class-weighted cross-entropy |
| Label smoothing | 0.05 |
| Early stopping patience | 10 epochs |
| Best checkpoint epoch | 6 |
| Best validation loss | 0.1716 |

### Augmentation (training only, stochastic per epoch)

| Augmentation | Probability | Parameters | Purpose |
|---|---|---|---|
| Background mixing | 70% (drone clips only) | SNR −5 to +20 dB | Primary robustness strategy — forces separation from environmental noise |
| Additive noise | 40% | SNR 10–30 dB | Mic noise floor and electrical interference |
| Pitch shift | 30% | ±2 semitones | Different drone models / RPMs / Doppler shift |
| Time stretch | 20% | 0.9×–1.1× | Temporal variation |
| Gain variation | 50% | −6 to +6 dB | Different mic sensitivities |
| SpecAugment | 50% | 2 freq masks (≤27 bins), 2 time masks (≤25 frames) | Robustness to partial occlusion |

### Real-Time Inference (in pipeline.py)

| Parameter | Value |
|---|---|
| Audio device | ReSpeaker 4 Mic Array (device 0) |
| Native sample rate | 16,000 Hz (no resampling needed) |
| Window duration | 1 second |
| Inference cadence | Every ~1 second |
| Probability smoothing | 3-window rolling average |
| Detection threshold | 0.5 (configurable via `--audio-threshold`) |
| Output | `acoustic_confidence` (0.0–1.0) in every WS event payload |

---

## Event Payload

Every detection event sent to the backend includes both model outputs:

```json
{
  "base_station_id": "543a8b49-fb4b-4e43-85f5-f60be027caf5",
  "drone_detected": true,
  "yolo_confidence": 0.847,
  "acoustic_confidence": 0.731,
  "image_url": "detections/2026-05-16T11-40-39_a1b2c3d4.jpg",
  "description": "[TRUE ALARM] A small quadcopter with visible rotors detected at mid-frame.",
  "detected_at": "2026-05-16T11:40:39.123Z"
}
```

| Field | Source |
|---|---|
| `drone_detected` | YOLO result, overridden to `false` if Gemini disagrees |
| `yolo_confidence` | Highest UAV box confidence in the triggering frame |
| `acoustic_confidence` | Latest 3-window smoothed CNN probability at event time |
| `image_url` | Supabase storage path — backend builds public URL from this |
| `description` | Gemini 2.0 Flash report, or `[event_type]` if `--no-gpt` |

### Event Types

| Event | Trigger |
|---|---|
| `drone_appeared` | 3 consecutive positive YOLO frames while state is CLEAR |
| `drone_ongoing` | Every 30 s while state remains DETECTED (heartbeat) |
| `drone_gone` | Suppressed — would produce imageless entries in dashboard |

---

## Gemini 2.0 Flash Verification

When `--no-gpt` is **not** set, each `drone_appeared` and `drone_ongoing` event is
sent to Gemini 2.0 Flash (via REST API, no SDK required) for visual confirmation.

- Gemini receives the JPEG frame as base64 and returns `{"is_drone": bool, "report": "..."}`
- If `is_drone: false`, the payload field `drone_detected` is set to `false`
- Rate-limited to one call per 15 s (free tier: 15 req/min)
- Requires `GEMINI_API_KEY` in `.env`

---

## WebRTC Live Stream

The pipeline maintains a persistent WebSocket to `/ws/webrtc/producer` on the backend.
When a dashboard viewer connects:

1. Backend sends `viewer_connected`
2. Pipeline creates a new `RTCPeerConnection` and attaches a fresh `YOLOVideoTrack`
3. SDP offer/answer and ICE candidates are exchanged via the signaling WebSocket
4. Annotated frames (bounding boxes + state overlay) stream peer-to-peer to the browser

STUN servers used: `stun.l.google.com:19302`, `stun1.l.google.com:19302`

---

## CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--no-show` | off | Headless mode — required when no display is connected |
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

## Environment Variables (`.env`)

```env
BACKEND_WS_URL=wss://se496-capstone-dashboard-backend.onrender.com/ws/detections
BASE_STATION_ID=543a8b49-fb4b-4e43-85f5-f60be027caf5
SUPABASE_URL=https://unjuiavhmurpcgcqnsby.supabase.co
SUPABASE_KEY=<service role key — NOT the anon key>
SUPABASE_BUCKET=detections
GEMINI_API_KEY=<optional — only needed without --no-gpt>
SIGNED_URL_EXPIRY_SECONDS=3600
```

> `SUPABASE_KEY` must be the **service role key**. The anon key blocks storage writes.

---

## Supabase Image Storage

Frames are encoded as JPEG (quality 85) and uploaded to the `detections` bucket.

```
Bucket:     detections
Path:       detections/<YYYY-MM-DDTHH-MM-SS>_<8-char-uid>.jpg
Public URL: https://<project>.supabase.co/storage/v1/object/public/detections/<path>
```

The backend constructs the full public URL from the `image_url` field in the payload.

---

## First-Time Setup

```bash
# 1. Install system libraries
sudo apt-get install -y libopenblas-dev libjpeg-dev zlib1g-dev libpython3-dev \
    libavcodec-dev libavformat-dev libswscale-dev cmake ninja-build

# 2. Install NVIDIA PyTorch + TensorRT + ultralytics (~60 min, builds torchvision from source)
bash setup_yolo_jetson.sh

# 3. Install pipeline dependencies
pip3 install -r requirements_pipeline.txt

# 4. Place model files:
#    - best.pt        → yolo_inference/best.pt
#    - audio best.pt  → yolo_inference/audio_detection/outputs/checkpoints/cnn_baseline/best.pt

# 5. Build TensorRT engine (~15 min, Jetson-specific)
python3 export_to_trt.py

# 6. Configure environment
cp .env.example .env   # fill in SUPABASE_KEY, BASE_STATION_ID, BACKEND_WS_URL

# 7. Run
PYTHONUNBUFFERED=1 python3 -u pipeline.py --no-show --no-gpt --audio-device 0
```

---

## Auto-Start with systemd

```bash
# Install and enable both services
sudo chmod +x install_services.sh
sudo ./install_services.sh

# Manage
sudo systemctl start   uscar-vision
sudo systemctl stop    uscar-vision
sudo systemctl status  uscar-vision
journalctl -u uscar-vision -f   # live logs
```

The `uscar-vision.service` runs the full pipeline (YOLO + audio + WebRTC + backend).
The `uscar-audio.service` is kept for standalone audio testing only.

---

## Files

| File / Directory | Purpose |
|---|---|
| `pipeline.py` | **Main entry point** — YOLO + AudioWorker + WebRTC + EventWorker |
| `export_to_trt.py` | Converts `best.pt` → `best.engine` (TensorRT FP16) |
| `setup_yolo_jetson.sh` | One-time environment setup for JetPack 5.1.x |
| `requirements_pipeline.txt` | pip dependencies for `pipeline.py` |
| `uscar-vision.service` | systemd unit — full pipeline on boot |
| `uscar-audio.service` | systemd unit — standalone audio pipeline |
| `install_services.sh` | Installs and enables both systemd services |
| `.env.jetson.example` | Template for environment variables |
| `audio_detection/audio_client.py` | Standalone audio pipeline (testing only) |
| `audio_detection/src/models/cnn_baseline.py` | CNNBaseline model definition |
| `audio_detection/src/data/features.py` | Log-mel spectrogram extraction |
| `audio_detection/configs/default.yaml` | Audio model hyperparameters |
| `audio_detection/DESIGN.md` | Full architecture decision rationale |
| `backend/` | FastAPI backend (deployed on Render) |

---

## Verified Package Stack

| Package | Version |
|---|---|
| torch | 2.1.0a0+41361538.nv23.06 (NVIDIA JetPack wheel) |
| torchvision | 0.16.1 (built from source) |
| numpy | 1.23.5 |
| opencv | 4.5.4 |
| tensorrt | 8.5.2.2 |
| ultralytics | 8.4.21 |
| sounddevice | 0.5.5 |
| librosa | 0.11.0 |
| websockets | 11.x |
| aiortc | 1.x |

---

## Troubleshooting

**`[AudioWorker] Failed to load — audio disabled`**
Place the audio checkpoint at `audio_detection/outputs/checkpoints/cnn_baseline/best.pt`.

**`[IMG] Upload failed: 403`**
`SUPABASE_KEY` is the anon key. Replace with the service role key.

**`[WS] WS connect failed`**
Check `BACKEND_WS_URL` in `.env`. Ensure the Render backend is running.

**Camera resolution is wrong**
Jetson GStreamer may ignore `cap.set()`. YOLO rescales to 640×640 regardless — this is harmless.

**WebRTC video is black**
Ensure `update_webrtc_frame()` is being called in the inference loop and a display is connected or `--no-show` is set.

**Audio not detecting**
Run `python3 pipeline.py --list-audio-devices` and pass the ReSpeaker index with `--audio-device 0`.

**Pipeline killed with exit code 144**
Killed by `pkill` — normal, not an error.

**TRT warning: `Using engine plan file across different models`**
Rebuild `best.engine` on this specific Jetson unit using `export_to_trt.py`.
