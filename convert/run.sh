#!/bin/bash
# Собирает Docker-образ и конвертирует ONNX → RKNN для RK3588.
# Результат: ../models/scrfd.rknn и ../models/arcface.rknn
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"

mkdir -p "$MODELS_DIR"

echo "=== Сборка Docker-образа (первый раз — долго, ~5-10 мин) ==="
docker build --platform linux/amd64 -t rknn-convert "$SCRIPT_DIR"

echo ""
echo "=== Конвертация моделей ==="
docker run --rm --platform linux/amd64 \
    -v "$MODELS_DIR:/output" \
    rknn-convert

echo ""
echo "=== Результат ==="
ls -lh "$MODELS_DIR"/*.rknn 2>/dev/null || echo "[!] Файлы .rknn не найдены"
