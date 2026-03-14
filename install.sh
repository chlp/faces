#!/bin/bash
set -e

echo "=== Установка системных зависимостей ==="

sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-venv \
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

echo "=== Установка Python-пакетов ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Установка rknn_toolkit_lite2 ==="
# rknnlite НЕ доступен через PyPI — нужен официальный .whl от Rockchip.
# Скачай нужный файл (Python-версию смотри через: python3 --version):
#   https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2/packages
# Пример: rknn_toolkit_lite2-2.3.0-cp310-cp310-linux_aarch64.whl
#
# Если положишь .whl рядом с этим скриптом — установится автоматически:
RKNN_WHL=$(ls rknn_toolkit_lite2-*linux_aarch64.whl 2>/dev/null | tail -1)
if [ -n "$RKNN_WHL" ]; then
    pip install "$RKNN_WHL"
    echo "[+] rknn_toolkit_lite2 установлен из $RKNN_WHL"
else
    echo ""
    echo "[!] rknn_toolkit_lite2 не найден рядом со скриптом."
    echo "    1. Скачай .whl с: https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2/packages"
    echo "    2. Положи рядом с install.sh и выполни:"
    echo "       source venv/bin/activate && pip install rknn_toolkit_lite2-*.whl"
fi

echo "=== Создание папок ==="
mkdir -p known_faces models

echo ""
echo "Следующий шаг — скачать RKNN модели:"
echo "  bash download_models.sh"
echo ""
echo "Запуск:"
echo "  source venv/bin/activate"
echo "  python3 main.py"
echo ""
echo "Добавление нового человека:"
echo "  mkdir -p known_faces/ИМЯ"
echo "  # положи фото в known_faces/ИМЯ/photo.jpg"
