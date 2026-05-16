"""
test_setup.py — Verify the audio pipeline dependencies on Jetson Orin NX.

Checks every import and runs a dry inference pass with a synthetic waveform.
Does NOT require the microphone or the model checkpoint.

Run from the audio_detection/ directory:
    cd /home/anis/yolo_inference/audio_detection
    python3 test_setup.py
"""

import sys
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent))

PASS = "[OK]"
FAIL = "[FAIL]"

def check(label, fn):
    try:
        result = fn()
        print(f"  {PASS}  {label}" + (f" — {result}" if result else ""))
        return True
    except Exception as e:
        print(f"  {FAIL}  {label} — {e}")
        return False

print("=" * 55)
print(" U-SCAR Audio Pipeline — Setup Verification")
print("=" * 55)
print()

failures = 0

# ---- Core deps ----
print("[ Dependencies ]")

ok = check("numpy import", lambda: None)
try:
    import numpy as np
    ver = np.__version__
    if not ver.startswith("1.23"):
        print(f"  [WARN] numpy is {ver} — expected 1.23.5 (vision env may be at risk)")
    else:
        print(f"  {PASS}  numpy version {ver} (correct)")
except Exception as e:
    print(f"  {FAIL}  numpy — {e}")
    failures += 1

if not check("torch import", lambda: __import__("torch").__version__): failures += 1
if not check("librosa import", lambda: __import__("librosa").__version__): failures += 1
if not check("sounddevice import", lambda: __import__("sounddevice").__version__): failures += 1
if not check("soundfile import", lambda: __import__("soundfile").__version__): failures += 1
if not check("yaml import", lambda: __import__("yaml").__version__): failures += 1

print()

# ---- Project imports ----
print("[ Project modules ]")

if not check("src.data.features", lambda: None):
    try:
        from src.data.features import extract_log_mel
        print(f"  {PASS}  src.data.features.extract_log_mel")
    except Exception as e:
        print(f"  {FAIL}  src.data.features — {e}")
        failures += 1
else:
    try:
        from src.data.features import extract_log_mel
        print(f"  {PASS}  src.data.features.extract_log_mel")
    except Exception as e:
        print(f"  {FAIL}  src.data.features — {e}")
        failures += 1

try:
    from src.models.cnn_baseline import CNNBaseline
    print(f"  {PASS}  src.models.cnn_baseline.CNNBaseline")
except Exception as e:
    print(f"  {FAIL}  src.models.cnn_baseline — {e}")
    failures += 1

try:
    from src.utils.config import load_config
    cfg = load_config("configs/default.yaml")
    print(f"  {PASS}  configs/default.yaml loaded")
except Exception as e:
    print(f"  {FAIL}  config loading — {e}")
    failures += 1

print()

# ---- Dry inference pass ----
print("[ Dry inference (no checkpoint needed) ]")

try:
    import numpy as np
    import torch
    from src.data.features import extract_log_mel
    from src.models.cnn_baseline import CNNBaseline
    from src.utils.config import load_config

    cfg      = load_config("configs/default.yaml")
    model    = CNNBaseline()
    model.eval()

    waveform = np.zeros(16000, dtype=np.float32)
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
    assert features.shape == (128, 101), f"Unexpected shape: {features.shape}"

    tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        prob = float(torch.softmax(logits, dim=-1)[0][1])

    print(f"  {PASS}  feature extraction shape: {features.shape}")
    print(f"  {PASS}  model forward pass OK — drone prob on silence: {prob:.4f}")
except Exception as e:
    print(f"  {FAIL}  dry inference — {e}")
    failures += 1

print()

# ---- Checkpoint ----
print("[ Checkpoint ]")
ckpt_path = Path("outputs/checkpoints/cnn_baseline/best.pt")
if ckpt_path.exists():
    size_mb = ckpt_path.stat().st_size / 1e6
    print(f"  {PASS}  checkpoint found ({size_mb:.1f} MB)")

    try:
        import torch
        from src.models.cnn_baseline import CNNBaseline
        model = CNNBaseline()
        ckpt  = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"  {PASS}  checkpoint loaded and weights verified")
    except Exception as e:
        print(f"  {FAIL}  checkpoint load failed — {e}")
        failures += 1
else:
    print(f"  [WAIT] checkpoint not found at {ckpt_path}")
    print( "         Place best.pt there once training completes.")

print()

# ---- Audio devices ----
print("[ Audio devices ]")
try:
    import sounddevice as sd
    devices = sd.query_devices()
    inputs  = [d for d in devices if d["max_input_channels"] > 0]
    if inputs:
        print(f"  {PASS}  {len(inputs)} input device(s) found:")
        for d in inputs:
            idx = list(devices).index(d)
            print(f"         [{idx}] {d['name']}  ({int(d['default_samplerate'])} Hz)")
    else:
        print("  [WARN] no input devices found — mic not connected yet (expected)")
except Exception as e:
    print(f"  {FAIL}  sounddevice query — {e}")
    failures += 1

print()
print("=" * 55)
if failures == 0:
    print(" All checks passed. Audio pipeline is ready.")
    print(" Next step: place best.pt and run audio_client.py")
else:
    print(f" {failures} check(s) failed. Fix the issues above before continuing.")
print("=" * 55)
