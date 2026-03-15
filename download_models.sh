#!/bin/bash
# Конвертирует ONNX-модели InsightFace в RKNN для RK3588 через Docker.
# Запускать на машине с Docker (Mac, Linux x86_64).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ -f "$SCRIPT_DIR/models/scrfd.rknn" ] && [ -f "$SCRIPT_DIR/models/arcface.rknn" ]; then
    echo "[+] Модели уже есть:"
    ls -lh "$SCRIPT_DIR/models/"*.rknn
    exit 0
fi

if ! command -v docker &>/dev/null; then
    echo "[!] Docker не найден."
    echo "    Установи Docker Desktop и повтори: bash download_models.sh"
    exit 1
fi

bash "$SCRIPT_DIR/convert/run.sh"
