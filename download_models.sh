#!/bin/bash
# Converts InsightFace ONNX models to RKNN for RK3588 via Docker.
# Run on a machine with Docker (Mac, Linux x86_64).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -f "$SCRIPT_DIR/models/scrfd.rknn" ] && [ -f "$SCRIPT_DIR/models/arcface.rknn" ]; then
    echo "[+] Models already exist:"
    ls -lh "$SCRIPT_DIR/models/"*.rknn
    exit 0
fi

if ! command -v docker &>/dev/null; then
    echo "[!] Docker not found."
    echo "    Install Docker Desktop and retry: bash download_models.sh"
    exit 1
fi

bash "$SCRIPT_DIR/convert/run.sh"
