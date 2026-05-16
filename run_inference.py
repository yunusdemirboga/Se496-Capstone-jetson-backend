"""
YOLO11m TensorRT inference on USB Arducam camera.

Usage:
    python3 run_inference.py [--camera INDEX] [--conf THRESHOLD]

    --camera  : /dev/video device index (default: 0, try 1 or 2 if wrong device)
    --conf    : confidence threshold 0.0-1.0 (default: 0.5)
    --imgsz   : inference image size (default: 640)
    --no-show : disable display window (useful for headless / performance testing)

Examples:
    python3 run_inference.py
    python3 run_inference.py --camera 1 --conf 0.4
    python3 run_inference.py --no-show    # prints FPS only, no display
"""

import argparse
import sys
import time

# Compatibility: restore np.bool alias removed in numpy 1.24 (needed by system TensorRT binding)
import numpy as np
if not hasattr(np, 'bool'):
    np.bool = np.bool_
if not hasattr(np, 'int'):
    np.int = np.int_
if not hasattr(np, 'float'):
    np.float = np.float_
if not hasattr(np, 'complex'):
    np.complex = np.complex_
if not hasattr(np, 'object'):
    np.object = np.object_
if not hasattr(np, 'str'):
    np.str = np.str_

# Disable ultralytics auto-update to prevent mid-run numpy downgrade
from ultralytics.utils import SETTINGS
SETTINGS.update({'sync': False})

import cv2

# ---- Args ----
parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=0, help="USB camera device index")
parser.add_argument("--conf",   type=float, default=0.5, help="Confidence threshold")
parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
parser.add_argument("--no-show", action="store_true", help="Disable display window")
args = parser.parse_args()

ENGINE_PATH = "/home/anis/yolo_inference/best.engine"

# ---- Verify engine exists ----
import os
if not os.path.exists(ENGINE_PATH):
    sys.exit(
        f"ERROR: Engine not found at {ENGINE_PATH}\n"
        "Run export_to_trt.py first."
    )

# ---- Load YOLO TensorRT engine ----
print("Loading TensorRT engine (first load takes ~20 seconds)...")
from ultralytics import YOLO
model = YOLO(ENGINE_PATH, task="detect")
# Warm up model so predictor/names are initialized
import numpy as np
_dummy = np.zeros((640, 640, 3), dtype=np.uint8)
model.predict(source=_dummy, device=0, verbose=False)
print(f"Model loaded. Classes: {model.names}")
print()

# ---- Open USB camera ----
print(f"Opening USB camera at /dev/video{args.camera}...")
cap = cv2.VideoCapture(args.camera)

if not cap.isOpened():
    # Try the next index in case device enumeration differs
    print(f"  /dev/video{args.camera} failed, trying {args.camera + 1}...")
    cap = cv2.VideoCapture(args.camera + 1)
    if not cap.isOpened():
        sys.exit(
            "ERROR: Cannot open camera. Check it is plugged in and try:\n"
            "  ls /dev/video*\n"
            "  python3 run_inference.py --camera <index>"
        )

# Set camera resolution - 1280x720 is a good balance for Arducam USB
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera opened: {actual_w}x{actual_h}")
print()
print("Running inference. Press 'q' to quit.")
print()

# ---- FPS tracking ----
fps        = 0.0
frame_count = 0
t_start    = time.time()

# ---- Inference loop ----
while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to read frame from camera.")
        break

    # Run TensorRT inference on the frame
    results = model.predict(
        source=frame,
        conf=args.conf,
        imgsz=args.imgsz,
        device=0,           # GPU
        verbose=False,
    )

    # ---- FPS calculation (rolling average over last 30 frames) ----
    frame_count += 1
    elapsed = time.time() - t_start
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        frame_count = 0
        t_start = time.time()

    # ---- Draw results and FPS ----
    annotated = results[0].plot()   # draws boxes + labels on frame

    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
    )

    # ---- Print detections to terminal ----
    boxes = results[0].boxes
    if boxes is not None and len(boxes):
        for box in boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            label  = model.names[cls_id]
            print(f"  Detected: {label} ({conf:.2f})")

    # ---- Display ----
    if not args.no_show:
        cv2.imshow("YOLO11m TensorRT - Arducam USB", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Quitting...")
            break
    else:
        # Headless: just print FPS periodically
        if frame_count == 0:
            print(f"FPS: {fps:.1f}")

cap.release()
cv2.destroyAllWindows()
print("Done.")
