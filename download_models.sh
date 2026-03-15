#!/bin/bash
# Скачивает RKNN модели для RK3588 из официального репозитория Rockchip.
# Запускать на Orange Pi после установки git.
set -e

REPO="https://github.com/airockchip/rknn_model_zoo.git"
mkdir -p models

echo "=== Клонирование rknn_model_zoo (sparse, только модели) ==="
TMP=$(mktemp -d)
git clone --depth=1 --filter=blob:none --sparse "$REPO" "$TMP"
(
    cd "$TMP"
    git sparse-checkout set \
        examples/SCRFD/model \
        examples/mobilefacenet/model
)

# Ищем файлы для RK3588: предпочитаем scrfd_10g, fallback на scrfd_2.5g
SCRFD_SRC=$(find "$TMP" -iname "*scrfd_10g*rk3588*.rknn" | head -1)
[ -z "$SCRFD_SRC" ] && SCRFD_SRC=$(find "$TMP" -iname "*scrfd*rk3588*.rknn" | head -1)
FACE_SRC=$(find "$TMP" \( -iname "*mobilefacenet*rk3588*.rknn" -o -iname "*arcface*rk3588*.rknn" \) | head -1)

if [ -n "$SCRFD_SRC" ]; then
    cp "$SCRFD_SRC" models/scrfd.rknn
    echo "[+] Детекция:     models/scrfd.rknn  ($(basename "$SCRFD_SRC"))"
else
    echo "[!] SCRFD модель не найдена автоматически."
    echo "    Открой: $REPO/tree/main/examples/SCRFD/model/"
    echo "    Скопируй нужный *rk3588*.rknn файл в: models/scrfd.rknn"
fi

if [ -n "$FACE_SRC" ]; then
    cp "$FACE_SRC" models/arcface.rknn
    echo "[+] Распознавание: models/arcface.rknn  ($(basename "$FACE_SRC"))"
else
    echo "[!] MobileFaceNet/ArcFace модель не найдена автоматически."
    echo "    Открой: $REPO/tree/main/examples/mobilefacenet/model/"
    echo "    Скопируй нужный *rk3588*.rknn файл в: models/arcface.rknn"
fi

rm -rf "$TMP"

echo ""
echo "Готово. Проверь наличие обоих файлов:"
echo "  ls -lh models/"
