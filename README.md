# U-SCAR Jetson Backend

UAV detection pipeline for the **NVIDIA Jetson Orin NX** (JetPack 5.1.3).
Runs two parallel inference pipelines — YOLO11 vision and CNN audio — and streams detection events to the U-SCAR dashboard backend via WebSocket. Supports live WebRTC video streaming to the dashboard browser.

---

## Hardware Requirements

| Component | Spec |
|---|---|
| SBC | NVIDIA Jetson Orin NX 8 GB |
| OS | JetPack 5.1.3 (Ubuntu 20.04, Python 3.8) |
| Camera | USB/CSI camera (tested at 1280×720) |
| Microphone | ReSpeaker USB mic array (or any sounddevice-compatible mic) |

---

## Repository Structure

```
.
├── pipeline.py                  # Vision inference pipeline (YOLO + WebRTC + event dispatch)
├── export_to_trt.py             # Converts best.pt → best.engine (TensorRT FP16)
├── requirements_pipeline.txt    # pip dependencies for pipeline.py
├── setup_yolo_jetson.sh         # Full environment setup (PyTorch, TRT, ultralytics)
├── install_services.sh          # Installs systemd services for auto-start on boot
├── uscar-vision.service         # systemd unit for the vision pipeline
├── uscar-audio.service          # systemd unit for the audio pipeline
├── .env.example                 # Template for environment variables
│
└── audio_detection/
    ├── audio_client.py          # Audio inference pipeline (CNN + WebSocket)
    ├── configs/
    │   └── default.yaml         # Audio model configuration (mel, augmentation, training)
    ├── requirements.txt         # pip dependencies for audio_client.py
    ├── outputs/
    │   └── checkpoints/
    │       └── cnn_baseline/
    │           └── best.pt      # Audio model checkpoint — place here manually (not in git)
    └── src/
        ├── data/                # Feature extraction, windowing, dataset, augmentation
        ├── models/              # CNNBaseline and PANNs-CNN14 model definitions
        ├── evaluation/          # Metrics, evaluator, report generation
        ├── training/            # Trainer loop and loss functions
        └── utils/               # Audio utilities and config loader
```

---

## Setup

### Step 1 — Clone the repository

```bash
git clone https://github.com/yunusdemirboga/Se496-Capstone-jetson-backend.git
cd Se496-Capstone-jetson-backend
```

### Step 2 — Install vision dependencies

Run the full setup script. This installs NVIDIA PyTorch (CUDA-enabled, aarch64), builds torchvision from source, installs ultralytics, and creates a numpy compatibility shim. **Takes 45–90 minutes on first run.**

```bash
bash setup_yolo_jetson.sh
```

Then install the pipeline-specific packages:

```bash
pip3 install -r requirements_pipeline.txt
```

Also install `aiortc` and `av` for WebRTC live streaming:

```bash
pip3 install aiortc av
```

### Step 3 — Install audio dependencies

```bash
cd audio_detection
pip3 install -r requirements.txt
pip3 install sounddevice  # for live microphone capture
cd ..
```

> **Note:** The audio `requirements.txt` lists `Python 3.10+` packages. On the Jetson (Python 3.8), install compatible versions manually if any fail:
> ```bash
> pip3 install torch torchaudio librosa soundfile numpy scipy pyyaml
> ```

### Step 4 — Place model files

**YOLO vision model** — place `best.pt` in the repo root:
```
Se496-Capstone-jetson-backend/
└── best.pt          ← your trained YOLO11 weights
```

**Audio model checkpoint** — place `best.pt` in:
```
audio_detection/outputs/checkpoints/cnn_baseline/best.pt
```

### Step 5 — Export YOLO to TensorRT

Run once after placing `best.pt`. Generates `best.engine` (FP16 TensorRT engine, ~40 MB). Takes 5–15 minutes.

```bash
python3 export_to_trt.py
```

### Step 6 — Configure environment variables

```bash
cp .env.example .env
nano .env
```

Fill in all values:

```env
# WebSocket URL of the U-SCAR dashboard backend
BACKEND_WS_URL=wss://your-backend.onrender.com/ws/detections

# UUID of this Jetson's base station record in the database
BASE_STATION_ID=your-base-station-uuid-here

# Supabase credentials for image upload
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key
SUPABASE_BUCKET=detections

# How long Supabase signed URLs remain valid (seconds)
SIGNED_URL_EXPIRY_SECONDS=3600

# OpenAI key for GPT-4o-mini detection reports (omit to skip GPT)
OPENAI_API_KEY=sk-...
```

> **Important:** `SUPABASE_KEY` must be the **service role key**, not the anon key. The anon key blocks storage writes.

---

## Running the Pipelines

### Vision pipeline (YOLO + WebRTC)

```bash
# Headless (no display connected — standard for Jetson)
PYTHONUNBUFFERED=1 python3 -u pipeline.py --no-show

# With GPT reports disabled (faster, no OpenAI cost)
PYTHONUNBUFFERED=1 python3 -u pipeline.py --no-show --no-gpt

# Dry run — no backend connection, prints events to terminal only
python3 pipeline.py --dry-run --no-show
```

**Vision pipeline CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--camera` | `0` | Camera device index |
| `--conf` | `0.5` | YOLO confidence threshold |
| `--imgsz` | `640` | Inference image size |
| `--no-show` | off | Disable OpenCV display window (required on headless Jetson) |
| `--no-gpt` | off | Skip GPT-4o-mini description generation |
| `--dry-run` | off | Run inference only, no backend connection |
| `--backend` | from `.env` | Override backend WebSocket URL |
| `--station-id` | from `.env` | Override base station UUID |

### Audio pipeline (CNN mic inference)

```bash
# List available audio input devices
cd audio_detection
python3 audio_client.py --list-devices

# Run with live microphone (device index from --list-devices)
python3 audio_client.py --device 0 \
    --backend wss://your-backend.onrender.com/ws/detections \
    --station-id your-base-station-uuid

# Test without microphone (simulated silent audio)
python3 audio_client.py --simulate
```

**Audio pipeline CLI flags:**

| Flag | Default | Description |
|---|---|---|
| `--device` | system default | sounddevice input device index |
| `--threshold` | `0.5` | Drone probability threshold (0.0–1.0) |
| `--checkpoint` | `outputs/checkpoints/cnn_baseline/best.pt` | Path to model checkpoint |
| `--simulate` | off | Use silent audio instead of mic (no hardware needed) |
| `--list-devices` | — | Print available audio input devices and exit |
| `--backend` | None | Backend WebSocket URL |
| `--station-id` | None | Base station UUID (required when `--backend` is set) |

---

## Auto-Start with systemd

To have both pipelines start automatically on boot:

### 1. Update service files with your username and repo path

Edit `uscar-vision.service` and `uscar-audio.service` — replace the hardcoded paths:

```ini
User=anis                                    # → your Linux username
WorkingDirectory=/home/anis/yolo_inference   # → your repo path
EnvironmentFile=/home/anis/yolo_inference/.env
ExecStart=/usr/bin/python3 /home/anis/yolo_inference/pipeline.py --no-show
```

### 2. Install and enable the services

```bash
chmod +x install_services.sh
sudo ./install_services.sh
```

### 3. Manage the services

```bash
# Start now (without rebooting)
sudo systemctl start uscar-vision
sudo systemctl start uscar-audio

# Check status
sudo systemctl status uscar-vision
sudo systemctl status uscar-audio

# View live logs
journalctl -u uscar-vision -f
journalctl -u uscar-audio -f

# Stop
sudo systemctl stop uscar-vision uscar-audio

# Disable auto-start
sudo systemctl disable uscar-vision uscar-audio
```

---

## How the Vision Pipeline Works

```
Camera frame (30 fps)
    │
    ▼
YOLO11 TensorRT (GPU)         ← FP16 inference on best.engine
    │  bounding boxes + confidence
    ▼
State Machine
    │  CLEAR → DETECTED: 3 consecutive positive frames
    │  DETECTED → CLEAR:  10 consecutive negative frames
    │  Heartbeat:          every 30 s while drone stays in scene
    ▼
Event Queue (thread-safe)
    │
    ├─ EventWorker thread
    │    ├─ Upload JPEG to Supabase Storage
    │    ├─ Call GPT-4o-mini (confirms detection, writes report)
    │    └─ Send JSON payload to backend /ws/detections
    │
    └─ WebRTC thread
         ├─ Persistent WebSocket to /ws/webrtc/producer (signaling relay)
         ├─ On viewer_connected: create RTCPeerConnection + SDP offer
         └─ YOLOVideoTrack.recv(): streams annotated frames to browser
```

### Detection event payload

```json
{
  "base_station_id": "uuid",
  "drone_detected": true,
  "yolo_confidence": 0.847,
  "acoustic_confidence": null,
  "image_url": "detections/2026-05-09T11-40-39_a1b2c3d4.jpg",
  "description": "[TRUE ALARM] GPT report text here.",
  "detected_at": "2026-05-09T11:40:39.123Z"
}
```

`image_url` is a Supabase storage path. The dashboard backend constructs the full public URL as:
`{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{image_url}`

---

## How the Audio Pipeline Works

```
Microphone (1-second windows @ 16 kHz)
    │
    ▼
Log-mel spectrogram (128 mels, n_fft=1024, hop=160)
    │
    ▼
CNNBaseline model (PyTorch)    ← checkpoint: outputs/checkpoints/cnn_baseline/best.pt
    │  drone probability (0.0–1.0)
    ▼
3-window rolling average
    │
    ▼
Debounce state machine (3 consecutive windows to flip state)
    │  UAV_APPEARED / UAV_LOST
    ▼
WebSocket → backend /ws/detections
```

---

## Troubleshooting

**`name 'JPEG_QUALITY' is not defined`**
The `JPEG_QUALITY` constant is missing from `pipeline.py`. Add `JPEG_QUALITY = 85` to the constants block near the top of the file.

**`[IMG] Upload failed: ... 403`**
`SUPABASE_KEY` is the anon key. Use the service role key instead.

**Camera opens at 3840×2160 instead of 1280×720**
The Jetson GStreamer driver may ignore the `cap.set()` resolution request. This is harmless — YOLO rescales internally to `--imgsz 640`.

**WebRTC shows "Live" but video is black**
Ensure only one `YOLOVideoTrack()` instance is created per peer connection and that `update_webrtc_frame()` is called every inference cycle. Do not reuse a single global track instance across connections — `pc.close()` stops all attached tracks.

**Audio model not found**
Place `best.pt` at `audio_detection/outputs/checkpoints/cnn_baseline/best.pt`. Run from the `audio_detection/` directory.

**Pipeline killed with exit code 144**
Exit code 144 = killed by `pkill`. This is normal, not an error.

**TRT engine load warning: `Using an engine plan file across different models`**
This is a TensorRT warning when the engine was built on a different Jetson unit. Rebuild the engine on the target device using `export_to_trt.py`.
