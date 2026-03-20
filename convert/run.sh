#!/bin/bash
# Builds Docker image and converts ONNX -> RKNN for RK3588.
# Result: ../models/scrfd.rknn and ../models/arcface.rknn
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"

mkdir -p "$MODELS_DIR"

echo "=== Building Docker image (first time is slow, ~5-10 min) ==="
docker build --platform linux/amd64 -t rknn-convert "$SCRIPT_DIR"

echo ""
echo "=== Converting models ==="
docker run --rm --platform linux/amd64 \
    -v "$MODELS_DIR:/output" \
    rknn-convert

echo ""
echo "=== Result ==="
ls -lh "$MODELS_DIR"/*.rknn 2>/dev/null || echo "[!] .rknn files not found"
