#!/bin/bash
set -e

echo "=== Установка зависимостей ==="

sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    espeak-ng

# v4l-utils нужен только для диагностики камеры (v4l2-ctl --list-devices)
# ставим отдельно, игнорируя held-статус пакета
sudo apt-get install -y --allow-change-held-packages v4l-utils || \
    echo "[!] v4l-utils не установлен (необязательно для работы)"

echo "=== Создание виртуального окружения ==="
python3 -m venv venv
source venv/bin/activate

echo "=== Установка Python-пакетов (dlib компилируется ~10-15 мин) ==="
pip install --upgrade pip
pip install dlib
pip install -r requirements.txt

echo "=== Создание папки для лиц ==="
mkdir -p known_faces

echo ""
echo "Готово! Запуск:"
echo "  source venv/bin/activate"
echo "  python3 main.py"
echo ""
echo "Добавление нового человека:"
echo "  mkdir -p known_faces/ИМЯ"
echo "  # положи фото в known_faces/ИМЯ/photo.jpg"
