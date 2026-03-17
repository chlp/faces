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
- `data/` — runtime data (auto-created)
  - `faces.db` — SQLite event log (snapshots as BLOBs, auto-pruned to 15)

### Classes in main.py

| Class | Role |
|---|---|
| `Config` | Dataclass with all tuning knobs. Populated from argparse + env vars (`FACE_PORT`, `FACE_THRESHOLD`, etc.) |
| `EventStore` | SQLite persistence for detection events + snapshot BLOBs. Thread-safe (WAL mode) |
| `WebServer` | `ThreadingHTTPServer` serving frames from memory (no disk I/O). Endpoints: `/frame.jpg`, `/detections.json`, `/snap/<id>.jpg`, `/health`, `/reload` |
| `FaceDB` | Holds `known_encodings` + `name_index`. Watches `known_faces/` mtime every 30s. Hot-reloads via encode thread pause/resume protocol |
| `FaceTracker` | State machine: streaks, pending stranger, greeting cooldowns. `update(face_results, frame) → list[event]` |
| `FaceDetector` / `FaceEncoder` | RKNN model wrappers pinned to NPU Core0 / Core1 |

### Thread model

| Thread | What it does | RKNN core |
|---|---|---|
| Main | Camera read → SCRFD detect → queue faces → dispatch events → draw | Core0 |
| Encode | Dequeue faces → ArcFace encode → identify → queue results | Core1 |
| FrameWriter | JPEG-encode current frame → store in WebServer memory buffer | — |
| HTTP (pool) | Serve requests from WebServer memory buffers | — |

**Hot-reload protocol**: main sends `"pause"` sentinel to encode queue → waits for `"paused"` → reloads FaceDB (both models from main thread) → sends `"resume"`

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

## CLI arguments / env vars

```bash
python3 main.py --port 8080 --threshold 0.45 --lang ru --camera -1 --data-dir data
python3 main.py --no-web --no-tts --no-debug --display
```

Env vars: `FACE_PORT`, `FACE_THRESHOLD`, `FACE_LANG`, `FACE_CAMERA`, `FACE_DATA_DIR` (override defaults, CLI takes priority).

## Web endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Web UI (HTML/JS) |
| `GET /frame.jpg` | Live JPEG frame (from memory, ~3fps) |
| `GET /detections.json` | Recent events JSON |
| `GET /snap/<id>.jpg` | Event snapshot (from SQLite BLOB) |
| `GET /health` | `{uptime_s, last_detection_ts, frame_jpeg_bytes}` |
| `GET /reload` | Trigger hot-reload of face database |
| `GET /clear` | Delete all events and snapshots from SQLite |
| `GET /debug/aligned.jpg` | Last aligned face (debug) |

## Tuning knobs (Config dataclass)

| Field | Default | CLI flag | Effect |
|---|---|---|---|
| `recognition_threshold` | 0.45 | `--threshold` | Cosine similarity cutoff (higher = stricter) |
| `stranger_min_score` | 0.30 | — | Min similarity to count as "stranger"; below = ignore |
| `greet_cooldown` | 10 | — | Seconds between greetings per person |
| `process_every_n` | 1 | — | Process every Nth frame |
| `confirm_frames` | 3 | — | Consecutive recognitions before greeting |
| `streak_min_interval` | 1/3 s | — | Min seconds between streak increments |
| `score_window` | 7 | — | Frames over which score is averaged |
| `web_event_cooldown` | 30 | — | Seconds before same face set re-triggers event |
| `stranger_confirm_delay` | 5 | — | Seconds to wait before recording stranger |
| `lang` | "ru" | `--lang` | "ru" or "en" |
| `_SCORE_THRESH` | 0.50 | — | Face detection confidence threshold |
| `_NMS_THRESH` | 0.40 | — | NMS IoU threshold |
| `_FRONTAL_THRESH` | 0.60 | — | Max nose offset (higher = allow more profile) |

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
