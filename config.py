"""Constants and configuration."""

import argparse
import os
from dataclasses import dataclass

import numpy as np

# ── Model constants ──────────────────────────────────────────────────────────
SCRFD_MODEL   = "models/scrfd.rknn"
ARCFACE_MODEL = "models/arcface.rknn"

GREETINGS      = {"ru": "Привет, {}!", "en": "Hello, {}!"}
UNKNOWN_LABEL  = "Stranger"
ESPEAK_ARGS    = {"ru": ["-v", "ru", "-s", "140"], "en": ["-v", "en"]}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

_SCRFD_INPUT  = 640
_STRIDES      = [8, 16, 32]
_NUM_ANCHORS  = 2
_SCORE_THRESH = 0.50
_NMS_THRESH   = 0.40
_FRONTAL_THRESH = 0.60

_ARCFACE_TPL = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041],
], dtype=np.float32)

FRAME_INTERVAL = 1 / 3.0
SNAPSHOTS_MAX  = 15
RELOAD_CHECK_S = 30.0
DEBUG_INTERVAL = 5.0


# ── Configuration ────────────────────────────────────────────────────────────
@dataclass
class Config:
    known_faces_dir: str = "known_faces"
    camera_index: int = -1
    camera_max_index: int = 10
    no_frame_reconnect_after: int = 10
    no_frame_reconnect_delay: int = 5
    web_port: int = 8080
    data_dir: str = "data"
    recognition_threshold: float = 0.45
    stranger_min_score: float = 0.30
    greet_cooldown: float = 10.0
    process_every_n: int = 1
    confirm_frames: int = 3
    streak_min_interval: float = 1 / 3.0
    score_window: int = 7
    web_event_cooldown: float = 30.0
    stranger_confirm_delay: float = 5.0
    lang: str = "ru"
    debug: bool = True
    show_display: bool = False
    no_tts: bool = False

    @property
    def db_path(self):
        return os.path.join(self.data_dir, "faces.db")


def parse_args() -> Config:
    e = os.environ.get
    p = argparse.ArgumentParser(description="Face recognition (RKNN / RK3588)")
    p.add_argument("--port", type=int, default=int(e("FACE_PORT", "8080")))
    p.add_argument("--threshold", type=float,
                   default=float(e("FACE_THRESHOLD", "0.45")))
    p.add_argument("--lang", default=e("FACE_LANG", "ru"), choices=["ru", "en"])
    p.add_argument("--camera", type=int, default=int(e("FACE_CAMERA", "-1")))
    p.add_argument("--data-dir", default=e("FACE_DATA_DIR", "data"))
    p.add_argument("--no-web", action="store_true")
    p.add_argument("--no-tts", action="store_true")
    p.add_argument("--no-debug", action="store_true")
    p.add_argument("--display", action="store_true")
    a = p.parse_args()
    return Config(
        web_port=0 if a.no_web else a.port,
        recognition_threshold=a.threshold,
        lang=a.lang,
        camera_index=a.camera,
        data_dir=a.data_dir,
        no_tts=a.no_tts,
        debug=not a.no_debug,
        show_display=a.display,
    )
