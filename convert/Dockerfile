FROM --platform=linux/amd64 python:3.10-slim

RUN apt-get update && apt-get install -y \
    wget libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Скачиваем wheel напрямую через GitHub LFS CDN (media.githubusercontent.com)
RUN wget -q -O /tmp/rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
    "https://raw.githubusercontent.com/airockchip/rknn-toolkit2/master/rknn-toolkit2/packages/x86_64/rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" && \
    pip install --no-cache-dir /tmp/rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl && \
    rm /tmp/rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

RUN pip install --no-cache-dir numpy "onnx==1.16.1"

WORKDIR /workspace
COPY convert.py .

CMD ["python", "convert.py"]
