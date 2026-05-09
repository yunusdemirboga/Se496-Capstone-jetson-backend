#!/bin/bash
# YOLO11 + TensorRT setup for Jetson Orin Nano / Orin NX - JetPack 5.1.3
# Run with: bash setup_yolo_jetson.sh
# Estimated time: 45-90 min (torchvision build takes the longest)

set -e  # Exit on any error

echo "=========================================="
echo " YOLO11 Jetson Setup - JetPack 5.1.3"
echo "=========================================="

# ---- Step 1: Upgrade pip ----
echo ""
echo "[1/7] Upgrading pip..."
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade setuptools wheel

# ---- Step 2: Install NVIDIA PyTorch for JetPack 5.1.x ----
# This is the CUDA-enabled PyTorch wheel built by NVIDIA for aarch64/JetPack 5.1.x
# Standard PyPI torch does NOT support CUDA on Jetson
echo ""
echo "[2/7] Downloading NVIDIA PyTorch 2.1.0 for JetPack 5.1.x..."
TORCH_WHEEL="torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl"
TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/${TORCH_WHEEL}"

cd /tmp
if [ ! -f "${TORCH_WHEEL}" ]; then
    wget -q --show-progress "${TORCH_URL}" -O "${TORCH_WHEEL}"
fi

echo "Installing OpenBLAS (required by PyTorch)..."
sudo apt-get install -y libopenblas-dev

echo "Installing PyTorch..."
python3 -m pip install "/tmp/${TORCH_WHEEL}"

# Verify torch + CUDA
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# ---- Step 3: Build torchvision from source ----
# NVIDIA does not provide a torchvision wheel; must build for JetPack 5.1.x
echo ""
echo "[3/7] Building torchvision 0.16.1 from source (this takes ~30-60 min)..."
sudo apt-get install -y \
    libjpeg-dev zlib1g-dev libpython3-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    cmake ninja-build

cd /tmp
if [ -d "torchvision" ]; then
    rm -rf torchvision
fi
git clone --depth 1 --branch v0.16.1 https://github.com/pytorch/vision torchvision
cd torchvision

export BUILD_VERSION=0.16.1
FORCE_CUDA=1 python3 setup.py install --user 2>&1 | tail -5

python3 -c "import torchvision; print('torchvision:', torchvision.__version__)"

# ---- Step 4: Install Python dependencies ----
echo ""
echo "[4/7] Installing Python dependencies..."

# Pin numpy to 1.23.5 - this is what the TensorRT engine expects.
# numpy 1.26+ is not available for Python 3.8 on aarch64, and 1.24+
# breaks the system TensorRT Python binding (np.bool removed).
python3 -m pip install "numpy==1.23.5"

python3 -m pip install \
    "matplotlib>=3.3.0" \
    "Pillow>=7.1.2" \
    "PyYAML>=5.3.1" \
    "requests>=2.23.0" \
    "scipy>=1.4.1" \
    "tqdm>=4.64.0" \
    "psutil" \
    "py-cpuinfo" \
    "thop>=0.1.1" \
    "pandas>=1.1.4" \
    "seaborn>=0.11.0"

# onnx + onnxslim are required for the PT -> TensorRT export step
python3 -m pip install "onnx>=1.12.0,<2.0.0" "onnxslim>=0.1.71"

# ---- Step 5: Install ultralytics (no-deps to protect our torch install) ----
echo ""
echo "[5/7] Installing ultralytics..."
python3 -m pip install ultralytics==8.4.21 --no-deps

# ---- Step 6: Create numpy._core compatibility shim ----
# best.pt was saved with numpy >= 1.26 which uses numpy._core internally.
# Python 3.8 on aarch64 only supports numpy up to 1.24.4, so we create
# a shim that redirects numpy._core -> numpy.core so torch.load() can
# deserialize the model weights correctly.
echo ""
echo "[6/7] Creating numpy._core compatibility shim..."

NUMPY_SITE=$(python3 -c "import site; print(site.getusersitepackages())")
SHIM_DIR="${NUMPY_SITE}/numpy/_core"
mkdir -p "${SHIM_DIR}"

cat > "${SHIM_DIR}/__init__.py" << 'SHIM'
# Compatibility shim: numpy._core -> numpy.core (for models saved with numpy >= 1.26)
import sys
import numpy.core as _core
import numpy.core.multiarray as _multiarray
import numpy.core.umath as _umath
import numpy.core.fromnumeric as _fromnumeric

from numpy.core import *

# Register all critical submodules so pickle can find numpy._core.X
sys.modules['numpy._core.multiarray'] = _multiarray
sys.modules['numpy._core.umath'] = _umath
sys.modules['numpy._core.fromnumeric'] = _fromnumeric

# Register any other submodule that might be referenced
for _attr in dir(_core):
    _mod_name = f'numpy._core.{_attr}'
    if _mod_name not in sys.modules:
        _obj = getattr(_core, _attr)
        if isinstance(_obj, type(sys)):
            sys.modules[_mod_name] = _obj

multiarray = _multiarray
umath = _umath
fromnumeric = _fromnumeric
SHIM

# Create individual submodule proxy files
for mod in multiarray umath fromnumeric numeric numerictypes defchararray records \
           memmap function_base getlimits shape_base einsumfunc overrides \
           _multiarray_umath _add_newdocs _dtype _exceptions _internal \
           _methods _type_aliases _ufunc_config arrayprint; do
    cat > "${SHIM_DIR}/${mod}.py" << SUBMOD
# Compatibility shim: numpy._core.${mod} -> numpy.core.${mod}
from numpy.core.${mod} import *
try:
    from numpy.core import ${mod} as _m
    import sys as _sys
    _this = _sys.modules[__name__]
    for _attr in dir(_m):
        if not hasattr(_this, _attr):
            try:
                setattr(_this, _attr, getattr(_m, _attr))
            except Exception:
                pass
except Exception:
    pass
SUBMOD
done

python3 -c "import numpy._core.multiarray; print('  numpy._core shim: OK')"

# ---- Step 7: Verify full installation ----
echo ""
echo "[7/7] Verifying installation..."
echo "=========================================="
python3 - <<'EOF'
import numpy as np
# Restore aliases removed in numpy 1.24 (needed by system TensorRT binding)
if not hasattr(np, 'bool'):   np.bool   = np.bool_
if not hasattr(np, 'int'):    np.int    = np.int_
if not hasattr(np, 'float'):  np.float  = np.float_

import torch
import torchvision
import numpy
import cv2
import tensorrt as trt
from ultralytics import YOLO

print(f"  torch:       {torch.__version__}")
print(f"  torchvision: {torchvision.__version__}")
print(f"  numpy:       {numpy.__version__}")
print(f"  opencv:      {cv2.__version__}")
print(f"  tensorrt:    {trt.__version__}")
print(f"  CUDA:        {torch.cuda.is_available()}")
print(f"  GPU:         {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print("")
print("All dependencies OK!")
EOF

echo ""
echo "=========================================="
echo " Setup complete!"
echo " Next step: python3 export_to_trt.py"
echo "=========================================="
