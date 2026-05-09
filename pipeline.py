"""
pipeline.py — U-SCAR Vision Detection Pipeline (Jetson Orin NX)

Runs YOLO11m TRT inference and applies a frame-confirmation state machine
to fire three event types:

  drone_appeared  — 3 consecutive positive frames while state is CLEAR
                    → captures frame, uploads to Supabase, calls GPT, sends to backend
  drone_ongoing   — every 30 s while state remains DETECTED (heartbeat, no image)
  drone_gone      — 10 consecutive negative frames while state is DETECTED

Events are processed by a background thread that owns its own asyncio loop,
so network I/O (upload, GPT, WebSocket) never stalls the inference loop.

Usage:
    python3 pipeline.py \\
        --backend ws://LAPTOP_IP:8000/ws/detections \\
        --station-id BASE_STATION_UUID

    python3 pipeline.py --dry-run   # no backend, prints events to terminal only
    python3 pipeline.py --no-gpt    # skip GPT description

Required env vars (or put them in .env in this directory):
    BACKEND_WS_URL        e.g. ws://192.168.1.50:8000/ws/detections
    BASE_STATION_ID       UUID of this device's base station record in the DB
    SUPABASE_URL          your Supabase project URL
    SUPABASE_KEY          Supabase anon key
    SUPABASE_BUCKET       storage bucket name
    OPENAI_API_KEY        OpenAI key (not needed with --no-gpt)
    SIGNED_URL_EXPIRY_SECONDS   signed URL lifetime in seconds (default 3600)
"""

import argparse
import asyncio
import fractions
import functools
import json
import os
import queue
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse, urlunparse

import av
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.rtcconfiguration import RTCConfiguration, RTCIceServer
from aiortc.sdp import candidate_from_sdp

# ── numpy compat patch — must be before ultralytics ──────────────────────────
import numpy as np
if not hasattr(np, 'bool'):    np.bool    = np.bool_
if not hasattr(np, 'int'):     np.int     = np.int_
if not hasattr(np, 'float'):   np.float   = np.float_
if not hasattr(np, 'complex'): np.complex = np.complex_
if not hasattr(np, 'object'):  np.object  = np.object_
if not hasattr(np, 'str'):     np.str     = np.str_

from ultralytics.utils import SETTINGS
SETTINGS.update({'sync': False})

import cv2
from dotenv import load_dotenv

load_dotenv()

# ── Tuneable constants ────────────────────────────────────────────────────────
ENGINE_PATH        = "/home/anis/yolo_inference/best.engine"
CONFIRM_FRAMES     = 3    # consecutive positive frames required to enter DETECTED
MISS_FRAMES        = 10   # consecutive negative frames required to return to CLEAR
HEARTBEAT_INTERVAL = 30   # seconds between drone_ongoing heartbeat events
WS_RECONNECT_DELAY = 5    # seconds to wait before retrying a failed WS connection
JPEG_QUALITY       = 85
SIGNED_URL_EXPIRY  = int(os.getenv("SIGNED_URL_EXPIRY_SECONDS", "3600"))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="U-SCAR Vision Detection Pipeline")
    p.add_argument("--camera",     type=int,   default=0,
                   help="Camera device index (default 0)")
    p.add_argument("--conf",       type=float, default=0.5,
                   help="YOLO confidence threshold (default 0.5)")
    p.add_argument("--imgsz",      type=int,   default=640,
                   help="Inference image size (default 640)")
    p.add_argument("--no-show",    action="store_true",
                   help="Disable display window (headless mode)")
    p.add_argument("--dry-run",    action="store_true",
                   help="Print events to terminal only, no backend connection")
    p.add_argument("--no-gpt",     action="store_true",
                   help="Skip GPT description generation")
    p.add_argument("--backend",    default=os.getenv("BACKEND_WS_URL"),
                   help="Backend WebSocket URL (overrides BACKEND_WS_URL env var)")
    p.add_argument("--station-id", default=os.getenv("BASE_STATION_ID"),
                   help="Base station UUID (overrides BASE_STATION_ID env var)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# YOLO
# ─────────────────────────────────────────────────────────────────────────────

def load_yolo():
    if not Path(ENGINE_PATH).exists():
        sys.exit(
            f"ERROR: TensorRT engine not found at {ENGINE_PATH}\n"
            "Run export_to_trt.py first."
        )
    print("Loading TensorRT engine (first load ~20 s)...")
    from ultralytics import YOLO
    model = YOLO(ENGINE_PATH, task="detect")
    # Warmup so predictor and model.names are initialised
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(source=dummy, device=0, verbose=False)
    print(f"Model loaded. Classes: {model.names}")
    return model


def get_uav_class_ids(model_names: dict) -> set:
    """
    Return the set of class IDs that represent a positive (UAV/drone) detection.
    Matches any class whose name contains 'uav', 'drone', 'aerial', or 'quadcopter'.
    If no such class is found (single-class model), all detections are treated as positive.
    """
    positive_kws = ("uav", "drone", "aerial", "quadcopter")
    ids = {
        cls_id for cls_id, name in model_names.items()
        if any(kw in name.lower() for kw in positive_kws)
        and not name.lower().startswith("not")
    }
    if not ids:
        print("  [WARN] No UAV class found by name — treating all detections as positive.")
        ids = set(model_names.keys())
    print(f"  UAV class IDs: {ids} → {[model_names[i] for i in sorted(ids)]}")
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# Camera
# ─────────────────────────────────────────────────────────────────────────────

def open_camera(index: int):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"  /dev/video{index} failed, trying {index + 1}...")
        cap = cv2.VideoCapture(index + 1)
        if not cap.isOpened():
            sys.exit("ERROR: Cannot open camera. Check connection and try --camera <index>.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera opened: {w}x{h}")
    return cap


# ─────────────────────────────────────────────────────────────────────────────
# Supabase Storage (plain HTTP — no supabase-py required on Jetson)
# ─────────────────────────────────────────────────────────────────────────────

def upload_frame(frame: np.ndarray) -> Tuple[str, str]:
    """
    Encode a BGR frame as JPEG and upload it to Supabase Storage via the REST API.

    Returns:
        file_path  — storage path stored in the database  (e.g. detections/2026-04-02T14-00-00_abc12345.jpg)
        signed_url — temporary URL valid for SIGNED_URL_EXPIRY seconds, used for GPT
    """
    import requests

    supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
    supabase_key = os.getenv("SUPABASE_KEY", "")
    bucket       = os.getenv("SUPABASE_BUCKET", "")
    if not all([supabase_url, supabase_key, bucket]):
        raise RuntimeError("SUPABASE_URL, SUPABASE_KEY, and SUPABASE_BUCKET must be set.")

    _, buf       = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    image_bytes  = buf.tobytes()
    ts           = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    short_uid    = str(uuid.uuid4())[:8]
    file_path    = f"detections/{ts}_{short_uid}.jpg"

    auth_headers = {"Authorization": f"Bearer {supabase_key}"}

    # Upload
    upload_resp = requests.post(
        f"{supabase_url}/storage/v1/object/{bucket}/{file_path}",
        headers={**auth_headers, "Content-Type": "image/jpeg"},
        data=image_bytes,
        timeout=15,
    )
    upload_resp.raise_for_status()

    # Request signed URL
    sign_resp = requests.post(
        f"{supabase_url}/storage/v1/object/sign/{bucket}/{file_path}",
        headers={**auth_headers, "Content-Type": "application/json"},
        json={"expiresIn": SIGNED_URL_EXPIRY},
        timeout=10,
    )
    sign_resp.raise_for_status()
    data        = sign_resp.json()
    signed_path = data.get("signedURL") or data.get("signedUrl") or ""
    # Supabase returns a relative path — prepend the project URL
    signed_url  = f"{supabase_url}{signed_path}" if signed_path.startswith("/") else signed_path

    return file_path, signed_url


# ─────────────────────────────────────────────────────────────────────────────
# GPT description
# ─────────────────────────────────────────────────────────────────────────────

def call_gpt(image_url: str, yolo_conf: float, timestamp: str) -> dict:
    """
    Send the captured frame (via signed URL) to GPT-4o-mini.

    Returns:
        {"is_drone": bool, "description": str}

    is_drone — True if GPT confirms a UAV/drone is present, False if false alarm.
    description — GPT's report of what it sees in the image.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return {"is_drone": True, "description": "GPT disabled — OPENAI_API_KEY not set."}
    try:
        from openai import OpenAI
        client   = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": (
                        f"This frame was captured by a UAV surveillance system at {timestamp} UTC. "
                        f"The YOLO vision model flagged a potential UAV with confidence {yolo_conf:.1%}. "
                        "Carefully examine the image. "
                        "Respond ONLY with a JSON object in this exact format: "
                        '{"is_drone": true or false, "report": "2-3 sentence description of what you see and whether a drone or UAV is actually present"}'
                    )},
                ],
            }],
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        return {
            "is_drone":    bool(data.get("is_drone", True)),
            "description": data.get("report", "No report returned."),
        }
    except Exception as e:
        return {"is_drone": True, "description": f"GPT error: {e}"}


# ─────────────────────────────────────────────────────────────────────────────
# Event Worker — background daemon thread with its own asyncio loop
# ─────────────────────────────────────────────────────────────────────────────

class EventWorker(threading.Thread):
    """
    Consumes detection events from a queue and handles all network operations:
      1. Image upload to Supabase Storage
      2. GPT description generation
      3. WebSocket send to backend /ws/detections

    Runs in a daemon thread so it is automatically killed when the main process exits.
    Maintains a persistent WebSocket connection and reconnects on drop.
    """

    def __init__(self, args, event_queue: queue.Queue):
        super().__init__(daemon=True, name="EventWorker")
        self.args        = args
        self.event_queue = event_queue
        self.ws          = None
        self.ws_ctx      = None

    # ── Thread entry point ────────────────────────────────────────────────────
    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._main())

    # ── Async main loop ───────────────────────────────────────────────────────
    async def _main(self):
        await self._connect()
        loop = asyncio.get_running_loop()
        try:
            while True:
                # Wait for the next event without blocking the asyncio loop.
                # queue.get(timeout=1) runs in a thread-pool executor so the
                # event loop stays alive for other coroutines (e.g. reconnect).
                try:
                    event = await loop.run_in_executor(
                        None,
                        functools.partial(self.event_queue.get, timeout=1.0),
                    )
                except queue.Empty:
                    continue
                await self._handle(event)
        except Exception as e:
            print(f"[EventWorker] Fatal error: {e}")
        finally:
            await self._close_ws()

    # ── WebSocket helpers ─────────────────────────────────────────────────────
    async def _connect(self):
        import websockets
        while True:
            try:
                self.ws_ctx = websockets.connect(self.args.backend)
                self.ws     = await self.ws_ctx.__aenter__()
                print("[EventWorker] WebSocket connected.")
                return
            except Exception as e:
                print(f"[EventWorker] WS connect failed: {e}  — retry in {WS_RECONNECT_DELAY}s")
                await asyncio.sleep(WS_RECONNECT_DELAY)

    async def _close_ws(self):
        if self.ws_ctx and self.ws:
            try:
                await self.ws_ctx.__aexit__(None, None, None)
            except Exception:
                pass

    # ── Event handler ─────────────────────────────────────────────────────────
    async def _handle(self, event: dict):
        event_type = event["type"]
        frame      = event.get("frame")       # np.ndarray or None
        yolo_conf  = event.get("conf")        # float or None
        now_str    = datetime.now(timezone.utc).isoformat()

        print(f"\n[EVENT] {event_type}  "
              f"conf={f'{yolo_conf:.3f}' if yolo_conf is not None else 'N/A'}")

        file_path   = None
        is_drone    = True   # default: trust YOLO unless GPT says otherwise
        description = None

        # ── drone_appeared / drone_ongoing: capture image, upload, GPT ──────────
        if event_type in ("drone_appeared", "drone_ongoing") and frame is not None:
            loop = asyncio.get_running_loop()

            try:
                file_path, signed_url = await loop.run_in_executor(
                    None, functools.partial(upload_frame, frame)
                )
                print(f"  [IMG] Uploaded  → {file_path}")
            except Exception as e:
                print(f"  [IMG] Upload failed: {e}")
                file_path  = None
                signed_url = None

            if not self.args.no_gpt and file_path:
                supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
                bucket       = os.getenv("SUPABASE_BUCKET", "")
                public_url   = f"{supabase_url}/storage/v1/object/public/{bucket}/{file_path}"
                try:
                    gpt_result  = await loop.run_in_executor(
                        None,
                        functools.partial(call_gpt, public_url, yolo_conf or 0.0, now_str),
                    )
                    is_drone    = gpt_result["is_drone"]
                    description = gpt_result["description"]
                    alarm_label = "TRUE ALARM" if is_drone else "FALSE ALARM"
                    print(f"  [GPT] {alarm_label} — {description}")
                except Exception as e:
                    print(f"  [GPT] Failed: {e}")

        # ── Build payload ─────────────────────────────────────────────────────
        drone_detected = is_drone
        alarm_label    = "TRUE ALARM" if is_drone else "FALSE ALARM"
        desc_field     = (
            f"[{alarm_label}] {description}" if description else f"[{event_type}]"
        )
        payload = {
            "base_station_id":     self.args.station_id,
            "drone_detected":      drone_detected,
            "yolo_confidence":     round(yolo_conf, 4) if yolo_conf is not None else None,
            "acoustic_confidence": None,
            "image_url":           file_path if file_path else None,
            "description":         desc_field,
            "detected_at":         now_str,
        }

        # ── Send via WebSocket, reconnect and retry once on failure ───────────
        for attempt in range(2):
            try:
                await self.ws.send(json.dumps(payload))
                ack = await asyncio.wait_for(self.ws.recv(), timeout=3.0)
                print(f"  [WS]  ack: {ack}")
                return
            except Exception as e:
                print(f"  [WS]  Send failed (attempt {attempt + 1}): {e}")
                if attempt == 0:
                    await self._close_ws()
                    await self._connect()


# ─────────────────────────────────────────────────────────────────────────────
# WebRTC — YOLOVideoTrack + signaling
# ─────────────────────────────────────────────────────────────────────────────

# Shared frame store — written by main thread, read by YOLOVideoTrack
_webrtc_frame_lock   = threading.Lock()
_webrtc_latest_frame: Optional[np.ndarray] = None
_webrtc_frame_count  = 0  # for debug prints


def update_webrtc_frame(frame: np.ndarray):
    """Called from the main inference loop each cycle."""
    global _webrtc_latest_frame, _webrtc_frame_count
    with _webrtc_frame_lock:
        _webrtc_latest_frame = frame.copy()
        _webrtc_frame_count += 1
        if _webrtc_frame_count % 30 == 0:
            print(f"[WebRTC] frame updated, shape: {frame.shape}")


class YOLOVideoTrack(VideoStreamTrack):
    """
    Single global instance shared across all RTCPeerConnections.
    Reads from _webrtc_latest_frame which the inference loop updates every cycle.
    """
    kind = "video"

    def __init__(self):
        super().__init__()
        self._recv_count = 0

    async def recv(self):
        pts, time_base = await self.next_timestamp()
        self._recv_count += 1
        with _webrtc_frame_lock:
            frame = _webrtc_latest_frame
        if self._recv_count % 30 == 0:
            print(f"[WebRTC] recv called, frame is None: {frame is None}")
        if frame is None:
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_frame           = av.VideoFrame.from_ndarray(img, format="rgb24")
        video_frame.pts       = pts
        video_frame.time_base = time_base
        return video_frame


async def run_webrtc_signaling(backend_url: str):
    """
    Maintains a persistent WebSocket connection to /ws/webrtc/producer.
    Handles SDP offer/answer and ICE candidate exchange.
    Runs in its own daemon thread event loop.

    A fresh YOLOVideoTrack is created per viewer connection so that
    pc.close() (which stops all attached tracks) never corrupts the
    track used by the next viewer.
    """
    import websockets

    parsed        = urlparse(backend_url)
    signaling_url = urlunparse(
        (parsed.scheme, parsed.netloc, "/ws/webrtc/producer", "", "", "")
    )
    ice_config = RTCConfiguration(
        iceServers=[
            RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
            RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
        ]
    )

    pc: Optional[RTCPeerConnection] = None

    async def close_pc():
        nonlocal pc
        if pc is not None:
            await pc.close()
            pc = None

    while True:
        try:
            print(f"[WebRTC] Connecting to {signaling_url} ...")
            async with websockets.connect(signaling_url) as ws:
                print("[WebRTC] Signaling connected.")

                async for raw in ws:
                    msg      = json.loads(raw)
                    msg_type = msg.get("type")

                    if msg_type == "viewer_connected":
                        await close_pc()
                        pc = RTCPeerConnection(ice_config)
                        # Fresh track per connection — pc.close() stops tracks,
                        # so reusing a global track would break subsequent viewers.
                        pc.addTrack(YOLOVideoTrack())

                        @pc.on("connectionstatechange")
                        async def on_connection_state_change():
                            state = pc.connectionState if pc else "closed"
                            print(f"[WebRTC] Connection state: {state}")
                            if state in ("failed", "disconnected", "closed"):
                                await close_pc()

                        @pc.on("icecandidate")
                        async def on_ice(candidate):
                            if candidate:
                                await ws.send(json.dumps({
                                    "type":          "ice-candidate",
                                    "candidate":     candidate.to_sdp(),
                                    "sdpMid":        candidate.sdpMid,
                                    "sdpMLineIndex": candidate.sdpMLineIndex,
                                }))

                        offer = await pc.createOffer()
                        await pc.setLocalDescription(offer)
                        await ws.send(json.dumps({
                            "type":    "offer",
                            "sdp":     pc.localDescription.sdp,
                            "sdpType": pc.localDescription.type,
                        }))
                        print("[WebRTC] Offer sent.")

                    elif msg_type == "answer" and pc is not None:
                        await pc.setRemoteDescription(
                            RTCSessionDescription(
                                sdp=msg["sdp"], type=msg["sdpType"]
                            )
                        )
                        print("[WebRTC] Remote description set.")

                    elif msg_type == "ice-candidate" and pc is not None:
                        raw_cand = msg.get("candidate", "")
                        if raw_cand.startswith("candidate:"):
                            raw_cand = raw_cand[len("candidate:"):]
                        try:
                            cand               = candidate_from_sdp(raw_cand)
                            cand.sdpMid        = msg.get("sdpMid")
                            cand.sdpMLineIndex = msg.get("sdpMLineIndex")
                            await pc.addIceCandidate(cand)
                        except Exception as e:
                            print(f"[WebRTC] Bad ICE candidate: {e}")

                    elif msg_type == "viewer_disconnected":
                        await close_pc()
                        print("[WebRTC] Viewer disconnected, PC closed.")

        except Exception as e:
            print(f"[WebRTC] Signaling error: {e}  — retry in {WS_RECONNECT_DELAY}s")

        await close_pc()
        await asyncio.sleep(WS_RECONNECT_DELAY)


# ─────────────────────────────────────────────────────────────────────────────
# Main inference loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if not args.dry_run:
        if not args.backend:
            sys.exit(
                "ERROR: Provide --backend or set BACKEND_WS_URL.\n"
                "Use --dry-run to run without a backend connection."
            )
        if not args.station_id:
            sys.exit(
                "ERROR: Provide --station-id or set BASE_STATION_ID.\n"
                "Create a base station in the backend first, then copy its UUID here."
            )

    model    = load_yolo()
    uav_ids  = get_uav_class_ids(model.names)
    cap      = open_camera(args.camera)
    ev_queue = queue.Queue()

    if not args.dry_run:
        worker = EventWorker(args, ev_queue)
        worker.start()
        print("[Main] EventWorker started.")
        webrtc_thread = threading.Thread(
            target=lambda: asyncio.run(run_webrtc_signaling(args.backend)),
            daemon=True,
            name="WebRTCSignaling",
        )
        webrtc_thread.start()
        print("[Main] WebRTC signaling thread started.\n")

    # ── State machine variables ───────────────────────────────────────────────
    state            = "CLEAR"
    confirm_count    = 0
    miss_count       = 0
    last_send_ts     = 0.0

    # ── FPS tracking ──────────────────────────────────────────────────────────
    fps         = 0.0
    frame_count = 0
    t_fps       = time.time()

    print(f"State machine: CONFIRM={CONFIRM_FRAMES} frames | "
          f"MISS={MISS_FRAMES} frames | HEARTBEAT={HEARTBEAT_INTERVAL}s")
    print("Running. Press 'q' to quit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame from camera.")
                break

            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=0,
                verbose=False,
            )

            # FPS (rolling 1-second window)
            frame_count += 1
            if (time.time() - t_fps) >= 1.0:
                fps         = frame_count / (time.time() - t_fps)
                frame_count = 0
                t_fps       = time.time()

            # ── Parse detections ──────────────────────────────────────────────
            # Any box whose class ID is in uav_ids counts as a positive frame.
            # Multiple boxes in one frame = one positive frame.
            # best_conf tracks the highest confidence among all UAV boxes.
            boxes     = results[0].boxes
            any_drone = False
            best_conf = 0.0

            if boxes is not None and len(boxes):
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    if cls_id in uav_ids:
                        any_drone = True
                        if conf > best_conf:
                            best_conf = conf

            # ── Filter results to UAV class only before plotting ──────────────
            if results[0].boxes is not None and len(results[0].boxes):
                cls_arr = results[0].boxes.cls.cpu().numpy().astype(int)
                uav_mask = np.isin(cls_arr, list(uav_ids))
                results[0].boxes.data = results[0].boxes.data[uav_mask]

            # ── Annotated frame (bounding boxes drawn by YOLO) ────────────────
            annotated = results[0].plot()

            # ── State machine ─────────────────────────────────────────────────
            now = time.time()

            if any_drone:
                miss_count     = 0
                confirm_count += 1

                if state == "CLEAR" and confirm_count >= CONFIRM_FRAMES:
                    # Transition: CLEAR → DETECTED
                    state         = "DETECTED"
                    confirm_count = 0

                    if args.dry_run:
                        print(f"[DRY-RUN] drone_appeared  conf={best_conf:.3f}")
                    else:
                        ev_queue.put({
                            "type":  "drone_appeared",
                            "frame": annotated,
                            "conf":  best_conf,
                        })
                    last_send_ts = now

                elif state == "DETECTED" and (now - last_send_ts) >= HEARTBEAT_INTERVAL:
                    # Heartbeat while drone stays in scene
                    if args.dry_run:
                        print(f"[DRY-RUN] drone_ongoing  conf={best_conf:.3f}")
                    else:
                        ev_queue.put({
                            "type":  "drone_ongoing",
                            "frame": annotated,
                            "conf":  best_conf,
                        })
                    last_send_ts = now

            else:
                confirm_count  = 0
                miss_count    += 1

                if state == "DETECTED" and miss_count >= MISS_FRAMES:
                    # Transition: DETECTED → CLEAR
                    state      = "CLEAR"
                    miss_count = 0

                    if args.dry_run:
                        print("[DRY-RUN] drone_gone")

            # ── Display ───────────────────────────────────────────────────────
            status_text = (
                f"FPS:{fps:.1f}  STATE:{state}  "
                f"confirm:{confirm_count}  miss:{miss_count}"
            )
            cv2.putText(
                annotated, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

            update_webrtc_frame(annotated)

            if not args.no_show:
                cv2.imshow("U-SCAR Vision Pipeline", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
