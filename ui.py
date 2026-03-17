"""UI utilities: font, face drawing, TTS, camera."""

import os
import subprocess
import threading

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import Config, ESPEAK_ARGS, UNKNOWN_LABEL


# ── Шрифт ────────────────────────────────────────────────────────────────────
def _load_ui_font(size: int = 20):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    ]:
        if os.path.exists(path):
            print(f"[*] Шрифт: {path}")
            return ImageFont.truetype(path, size)
    print("[!] TTF-шрифт не найден, кириллица может не отображаться")
    return ImageFont.load_default()


_UI_FONT = _load_ui_font()


# ── TTS ──────────────────────────────────────────────────────────────────────
def speak(text: str, lang: str = "ru"):
    def _run():
        try:
            subprocess.run(
                ["espeak-ng"] + ESPEAK_ARGS.get(lang, []) + [text],
                check=True, capture_output=True,
            )
        except FileNotFoundError:
            print(f"[!] espeak-ng не установлен: {text}")
        except subprocess.CalledProcessError as e:
            print(f"[!] TTS: {e}")
    threading.Thread(target=_run, daemon=True).start()


# ── Отрисовка рамок ─────────────────────────────────────────────────────────
def draw_faces(frame, detected):
    """Рисует рамки и подписи сбоку (слева или справа, в зависимости от края)."""
    if not detected:
        return
    h_frame, w_frame = frame.shape[:2]
    for bbox, name, score in detected:
        x1, y1, x2, y2 = map(int, bbox)
        color = (0, 200, 0) if name != UNKNOWN_LABEL else (0, 0, 200)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        bbox_left = _UI_FONT.getlength(name)
        label_w = int(bbox_left) + 12
        label_h = 26
        if x2 + label_w <= w_frame:
            lx1 = x2
        else:
            lx1 = x1 - label_w
            if lx1 < 0:
                lx1 = 0
        ly1 = max(y1, 0)
        lx2 = lx1 + label_w
        ly2 = min(ly1 + label_h, h_frame)
        if lx2 > w_frame:
            lx2 = w_frame
        if lx2 <= lx1 or ly2 <= ly1:
            continue
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, cv2.FILLED)
        roi = frame[ly1:ly2, lx1:lx2]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(roi_rgb)
        draw = ImageDraw.Draw(pil)
        draw.text((6, 3), name, font=_UI_FONT, fill=(255, 255, 255))
        frame[ly1:ly2, lx1:lx2] = cv2.cvtColor(
            np.array(pil), cv2.COLOR_RGB2BGR
        )


# ── Камера ───────────────────────────────────────────────────────────────────
def open_camera(cfg: Config):
    indices = (
        [cfg.camera_index]
        if cfg.camera_index >= 0
        else range(cfg.camera_max_index)
    )
    for idx in indices:
        c = cv2.VideoCapture(idx)
        if c.isOpened():
            r, _ = c.read()
            if r:
                c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"[*] Камера idx={idx}: {w}x{h}")
                return c, idx
        c.release()
    return None, -1
