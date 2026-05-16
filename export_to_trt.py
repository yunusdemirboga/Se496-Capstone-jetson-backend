"""
Export YOLO11m .pt model to TensorRT .engine file.
Run AFTER setup_yolo_jetson.sh completes.

Usage:
    python3 export_to_trt.py

Output:
    /home/anis/Downloads/best.engine  (FP16 TensorRT engine)
"""

import os
import sys

# Verify environment before attempting export
print("Checking environment...")
try:
    import torch
    assert torch.cuda.is_available(), "CUDA not available - cannot build TRT engine"
    print(f"  torch     : {torch.__version__}")
    print(f"  GPU       : {torch.cuda.get_device_name(0)}")
    print(f"  CUDA      : {torch.version.cuda}")
except ImportError:
    sys.exit("ERROR: PyTorch not installed. Run setup_yolo_jetson.sh first.")

try:
    import tensorrt as trt
    print(f"  TensorRT  : {trt.__version__}")
except ImportError:
    sys.exit("ERROR: TensorRT Python binding not found.")

try:
    from ultralytics import YOLO
    print(f"  ultralytics: OK")
except ImportError:
    sys.exit("ERROR: ultralytics not installed. Run setup_yolo_jetson.sh first.")

print()

# ---- Paths ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PT = os.path.join(SCRIPT_DIR, "best.pt")
ENGINE_OUT = os.path.join(SCRIPT_DIR, "best.engine")

if not os.path.exists(MODEL_PT):
    sys.exit(f"ERROR: Model not found at {MODEL_PT}")

print(f"Model  : {MODEL_PT}  ({os.path.getsize(MODEL_PT) / 1e6:.1f} MB)")
print(f"Output : {ENGINE_OUT}")
print()

# ---- Load model and inspect ----
print("Loading model (this may take a moment)...")
model = YOLO(MODEL_PT)

print(f"  Task      : {model.task}")
print(f"  Classes   : {len(model.names)} -> {model.names}")
imgsz = model.model.args.get("imgsz", 640) if hasattr(model.model, "args") else 640
print(f"  Image size: {imgsz}")
print()

# ---- Export to TensorRT ----
# FP16 halves memory and doubles throughput on Jetson Orin
# workspace=4  --> 4 GB TRT workspace (safe for 8 GB device)
# batch=1      --> single-frame inference (adjust if needed)
print("Exporting to TensorRT FP16 engine...")
print("This takes 5-15 minutes on first run (TRT calibration).")
print()

model.export(
    format="engine",
    device=0,          # GPU 0 (the Jetson's integrated GPU)
    half=True,         # FP16 precision
    workspace=4,       # GB of TRT workspace
    batch=1,
    imgsz=imgsz,
    simplify=True,     # Simplify ONNX graph before TRT conversion
    verbose=False,
)

if os.path.exists(ENGINE_OUT):
    size_mb = os.path.getsize(ENGINE_OUT) / 1e6
    print()
    print(f"SUCCESS: TensorRT engine saved to {ENGINE_OUT}  ({size_mb:.1f} MB)")
    print()
    print("Next step: run  python3 run_inference.py")
else:
    # ultralytics saves engine next to the .pt file
    candidate = MODEL_PT.replace(".pt", ".engine")
    if os.path.exists(candidate):
        print(f"Engine saved at: {candidate}")
        print("Next step: run  python3 run_inference.py")
    else:
        print("WARNING: Engine file not found at expected path.")
        print("Check the ultralytics output above for the actual save location.")
