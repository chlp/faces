# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A minimal face recognition app for Orange Pi 5 Max (ARM64, Ubuntu/Debian). A USB webcam watches the apartment entrance; when a known person appears, the app greets them by name via espeak-ng TTS. Inference runs on the RK3588 NPU.

## Hardware target

- **Board**: Orange Pi 5 Max (RK3588, ARM64)
- **Camera**: Anker PowerConf C200 — USB, shows up as `/dev/video0`
- **OS**: Ubuntu/Debian ARM64

## Setup & run

```bash
# 1. Системные зависимости + venv + Python-пакеты
bash install.sh

# 2. Скачать rknn_toolkit_lite2 (.whl) вручную:
#    https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2/packages
#    положить рядом, затем:
source venv/bin/activate
pip install rknn_toolkit_lite2-*.whl

# 3. Скачать RKNN-модели
bash download_models.sh

# 4. Запуск
python3 main.py
```

## Adding people to the database

```bash
mkdir -p known_faces/NAME
# place one or more photos as known_faces/NAME/photo.jpg
```

Multiple photos per person improve accuracy. Face must be clearly visible.

## Key architecture

- `main.py` — single-file app, all logic here
- `known_faces/` — face database, one subfolder per person
- `models/` — RKNN model files (`scrfd.rknn`, `arcface.rknn`)

### NPU pipeline

| Stage | Model | Input | Output |
|---|---|---|---|
| Face detection | SCRFD-10G | 640×640, BGR→RGB, (x−127.5)/128 | 9 tensors (score/bbox/kps × 3 strides) |
| Face encoding | ArcFace ResNet100 (Glint360K) | 112×112 aligned, BGR→RGB, (x−127.5)/128 | 512-dim embedding |

**Detection post-processing** (`_decode_scrfd`): anchor-based decoding across 3 strides (8/16/32), 2 anchors per cell, followed by NMS. Anchor centers are pre-computed in `_ANCHORS`. Bbox predictions are distances (lt/rb) in stride units; keypoints are offsets in stride units.

**Recognition**: cosine similarity between L2-normalized embeddings (`known_encodings @ encoding`). Threshold: `RECOGNITION_THRESHOLD` (default 0.45). Score is averaged over last `SCORE_WINDOW` frames for stability.

**Face alignment**: 5-point affine transform (`cv2.estimateAffinePartial2D`) to the standard ArcFace 112×112 template `_ARCFACE_TPL`. Sideways profiles are skipped (`_FRONTAL_THRESH`).

- Detection runs on every `PROCESS_EVERY_N`-th frame; encoding runs async on NPU Core1 in parallel
- Greeting requires `CONFIRM_FRAMES` consecutive recognitions above threshold
- Greeting cooldown (`GREET_COOLDOWN` seconds) prevents repeated greetings per person
- TTS runs in a daemon thread via `subprocess` → `espeak-ng`
- Stranger events are delayed by `STRANGER_CONFIRM_DELAY` seconds (in case the person is identified before that)
- Live web stream freezes on the stranger's last frame until a known person appears

## Tuning knobs (top of main.py)

| Variable | Default | Effect |
|---|---|---|
| `CAMERA_MAX_INDEX` | 10 | Перебираются индексы 0..N−1 в поисках первой доступной камеры |
| `NO_FRAME_RECONNECT_AFTER` | 10 | Подряд «нет кадра» до переподключения |
| `NO_FRAME_RECONNECT_DELAY` | 5 | Секунд ожидания перед переподключением |
| `RECOGNITION_THRESHOLD` | 0.45 | Cosine similarity cutoff (higher = stricter) |
| `STRANGER_MIN_SCORE` | 0.30 | Min similarity to count as "stranger"; below = likely false detection, ignore |
| `GREET_COOLDOWN` | 10 | Seconds between greetings per person |
| `PROCESS_EVERY_N` | 1 | Process every Nth frame (1 = every frame) |
| `CONFIRM_FRAMES` | 3 | Consecutive recognitions required before greeting |
| `STREAK_MIN_INTERVAL` | 1/3 s | Min seconds between streak increments (3 streaks ≈ 1 s) |
| `SCORE_WINDOW` | 7 | Frames over which recognition score is averaged |
| `WEB_EVENT_COOLDOWN` | 30 | Seconds before the same face set triggers another web snapshot |
| `STRANGER_CONFIRM_DELAY` | 5 | Seconds to wait before recording a stranger event |
| `LANG` | "ru" | "ru" or "en" |
| `_SCORE_THRESH` | 0.50 | Face detection confidence threshold |
| `_NMS_THRESH` | 0.40 | NMS IoU threshold |
| `_FRONTAL_THRESH` | 0.60 | Max nose-from-eye-center offset (fraction of eye distance); higher = allow more profile |

## Dependency notes

`rknn_toolkit_lite2` is **not on PyPI** — download the matching `.whl` from:
https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2/packages

RKNN model files come from:
https://github.com/airockchip/rknn_model_zoo (handled by `download_models.sh`)

`opencv-python-headless` is used intentionally for ARM (no bundled Qt/GTK), but `cv2.imshow` still requires a connected display or X server.

## Troubleshooting SCRFD output format

If faces are not detected, the model may use a different tensor ordering or normalization. Check:
1. Print `[o.shape for o in outputs]` in `FaceDetector.detect()` to verify tensor sizes
2. Expected sizes at 640×640: score tensors with 12800 / 3200 / 800 elements
3. If bbox distances don't need stride multiplication, remove `* stride` in `_decode_scrfd`
