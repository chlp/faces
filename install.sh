#!/bin/bash
set -e

echo "=== Installing system dependencies ==="

sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    libx11-dev \
    libgtk-3-dev \
    espeak-ng

# v4l-utils is only needed for camera diagnostics (v4l2-ctl --list-devices)
# installed separately, ignoring held package status
sudo apt-get install -y --allow-change-held-packages v4l-utils || \
    echo "[!] v4l-utils not installed (optional)"

echo "=== Creating virtual environment ==="
python3 -m venv venv
source venv/bin/activate

echo "=== Installing Python packages ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Installing rknn_toolkit_lite2 ==="
# rknnlite is NOT available on PyPI — you need the official .whl from Rockchip.
# Download the matching file (check Python version via: python3 --version):
#   https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2/packages
# Example: rknn_toolkit_lite2-2.3.0-cp310-cp310-linux_aarch64.whl
#
# If you place the .whl next to this script, it will be installed automatically:
RKNN_WHL=$(ls rknn_toolkit_lite2-*linux_aarch64.whl 2>/dev/null | tail -1)
if [ -n "$RKNN_WHL" ]; then
    pip install "$RKNN_WHL"
    echo "[+] rknn_toolkit_lite2 installed from $RKNN_WHL"
else
    echo ""
    echo "[!] rknn_toolkit_lite2 not found next to the script."
    echo "    1. Download .whl from: https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2/packages"
    echo "    2. Place next to install.sh and run:"
    echo "       source venv/bin/activate && pip install rknn_toolkit_lite2-*.whl"
fi

echo "=== Updating librknnrt ==="
# System library may be outdated (incompatible with toolkit2 2.3.x models).
# Download the latest version from Rockchip directly.
LIBRKNN_URL="https://raw.githubusercontent.com/airockchip/rknn-toolkit2/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so"
if wget -q --spider "$LIBRKNN_URL" 2>/dev/null; then
    wget -q -O /tmp/librknnrt.so "$LIBRKNN_URL"
    sudo cp /tmp/librknnrt.so /usr/lib/librknnrt.so
    sudo ldconfig
    echo "[+] librknnrt updated"
else
    echo "[!] Failed to download librknnrt (no network?). Skipping."
fi

echo "=== Creating directories ==="
mkdir -p known_faces models

echo ""
echo "Next step — copy models to models/:"
echo "  models/scrfd.rknn and models/arcface.rknn"
echo "  (convert on Mac: bash convert/run.sh, then scp)"
echo ""
echo "Run:"
echo "  source venv/bin/activate"
echo "  python3 main.py"
echo ""
echo "Adding a new person:"
echo "  mkdir -p known_faces/NAME"
echo "  # place photos in known_faces/NAME/photo.jpg"
