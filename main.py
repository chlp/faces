#!/usr/bin/env python3
"""
Распознавание лиц на NPU (RKNN) — Orange Pi 5 Max / RK3588.
Детекция: SCRFD-10G  |  Энкодинг: ArcFace ResNet100 (Glint360K)
Скачать модели: bash download_models.sh
"""

import argparse
import http.server
import json
import math
import os
import queue
import signal
import sqlite3
import sys
import time
import subprocess
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rknnlite.api import RKNNLite

# ── Константы моделей ────────────────────────────────────────────────────────
SCRFD_MODEL   = "models/scrfd.rknn"
ARCFACE_MODEL = "models/arcface.rknn"

GREETINGS      = {"ru": "Привет, {}!", "en": "Hello, {}!"}
UNKNOWN_LABEL  = "Незнакомец"
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


# ── Конфигурация ─────────────────────────────────────────────────────────────
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


def _parse_args() -> Config:
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


# ── Шрифт / якоря ───────────────────────────────────────────────────────────
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


def _build_anchor_centers(input_size=_SCRFD_INPUT):
    centers = {}
    for stride in _STRIDES:
        fs = math.ceil(input_size / stride)
        gy, gx = np.mgrid[0:fs, 0:fs]
        c = np.stack([gx * stride, gy * stride], axis=-1).reshape(-1, 2)
        centers[stride] = np.repeat(c, _NUM_ANCHORS, axis=0).astype(np.float32)
    return centers


_ANCHORS = _build_anchor_centers()


# ── EventStore (SQLite) ─────────────────────────────────────────────────────
class EventStore:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.Lock()
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                names TEXT NOT NULL,
                snapshot BLOB
            )
        """)
        self._conn.commit()

    def add(self, names: list, jpeg_bytes: bytes) -> dict:
        ts = round(time.time(), 2)
        names_json = json.dumps(sorted(names), ensure_ascii=False)
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO events (ts, names, snapshot) VALUES (?, ?, ?)",
                (ts, names_json, jpeg_bytes),
            )
            eid = cur.lastrowid
            self._conn.execute(
                "DELETE FROM events WHERE id NOT IN "
                "(SELECT id FROM events ORDER BY id DESC LIMIT ?)",
                (SNAPSHOTS_MAX,),
            )
            self._conn.commit()
        return {"ts": ts, "names": sorted(names), "img": eid}

    def recent(self, limit: int = SNAPSHOTS_MAX) -> list:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, ts, names FROM events ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"ts": r[1], "names": json.loads(r[2]), "img": r[0]}
            for r in reversed(rows)
        ]

    def get_snapshot(self, event_id: int) -> bytes | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT snapshot FROM events WHERE id = ?", (event_id,)
            ).fetchone()
        return row[0] if row else None

    def last_greeted_times(self, window_s: float) -> dict:
        cutoff = time.time() - window_s
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, names FROM events WHERE ts > ? ORDER BY ts",
                (cutoff,),
            ).fetchall()
        result = {}
        for ts, names_json in rows:
            for name in json.loads(names_json):
                if name != UNKNOWN_LABEL:
                    result[name] = ts
        return result

    def close(self):
        self._conn.close()


# ── Веб: HTML ────────────────────────────────────────────────────────────────
_HTML = """<!DOCTYPE html>
<html lang="ru"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Камера</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:#0a0a0a;color:#e0e0e0;
  font-family:-apple-system,BlinkMacSystemFont,sans-serif}
body{display:grid;grid-template-rows:auto 1fr;
  padding:10px;padding-top:max(10px,env(safe-area-inset-top));gap:10px;overflow:hidden}
header{display:flex;align-items:center;gap:8px}
#dot{width:8px;height:8px;border-radius:50%;background:#555;transition:background .3s;flex-shrink:0}
#dot.live{background:#30d158}
header span{font-size:.75rem;font-weight:600;color:#555;letter-spacing:.08em;text-transform:uppercase}
#rbtn{margin-left:auto;background:none;border:1px solid #333;color:#888;
  border-radius:4px;padding:2px 8px;cursor:pointer;font-size:.8rem}
#rbtn:active{opacity:.5}
#grid{display:grid;grid-template-columns:repeat(4,1fr);grid-template-rows:repeat(4,1fr);
  gap:8px;min-height:0}
.cell{background:#1c1c1e;border-radius:10px;overflow:hidden;display:flex;flex-direction:column;min-height:0}
.cell img{width:100%;flex:1;object-fit:contain;display:block;min-height:0;background:#000}
.info{padding:5px 7px;flex-shrink:0;background:#1c1c1e}
.names{font-size:.75rem;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.names.k{color:#30d158}.names.u{color:#ff453a}.names.empty{color:#333}
.meta{font-size:.65rem;color:#555;margin-top:1px;display:flex;gap:6px}
.count{color:#888}
#live-label{font-size:.65rem;color:#555;padding:4px 7px;flex-shrink:0;text-align:center}
</style></head><body>
<header>
  <span id="dot"></span><span>Камера</span>
  <button id="rbtn" onclick="fetch('/reload').then(r=>r.json()).then(d=>{this.textContent='OK';setTimeout(()=>this.textContent='↻',2000)}).catch(()=>this.textContent='✗')">↻</button>
</header>
<div id="grid">
  <div class="cell" id="cell-live">
    <img id="cam" alt="">
    <div id="live-label">прямой эфир</div>
  </div>
</div>
<script>
const grid=document.getElementById('grid'),dot=document.getElementById('dot');
const TOTAL=16;
let lastTs=0,cells=[];
for(let i=0;i<TOTAL-1;i++){
  const c=document.createElement('div');
  c.className='cell';
  c.innerHTML='<img src="" alt="" style="visibility:hidden"><div class="info"><div class="names empty">—</div><div class="meta"><span class="count"></span></div></div>';
  grid.appendChild(c);
  cells.push(c);
}
function fmtDate(ts){
  const d=new Date(ts*1000);
  const dd=String(d.getDate()).padStart(2,'0');
  const mm=String(d.getMonth()+1).padStart(2,'0');
  const hh=String(d.getHours()).padStart(2,'0');
  const mi=String(d.getMinutes()).padStart(2,'0');
  const ss=String(d.getSeconds()).padStart(2,'0');
  return dd+'.'+mm+' '+hh+':'+mi+':'+ss;
}
function rebuild(evs){
  const list=evs.slice().reverse();
  cells.forEach((c,i)=>{
    const ev=list[i];
    const img=c.querySelector('img');
    const nm=c.querySelector('.names');
    const meta=c.querySelector('.meta');
    if(!ev){
      img.style.visibility='hidden';img.src='';
      nm.className='names empty';nm.textContent='—';
      meta.innerHTML='';
      return;
    }
    img.style.visibility='';
    if(img.dataset.ts!==String(ev.ts)){
      img.dataset.ts=ev.ts;
      img.src='snap/'+ev.img+'.jpg?t='+ev.ts;
    }
    const kn=ev.names.filter(n=>n!=='Незнакомец');
    const un=ev.names.filter(n=>n==='Незнакомец').length;
    const total=ev.names.length;
    const hasKnown=kn.length>0;
    let label=kn.join(', ');
    if(un>0)label+=(label?'\u00a0+\u00a0':'')+(un===1?'незн.':un+'\u00a0незн.');
    nm.className='names '+(hasKnown?'k':'u');
    nm.textContent=label||'Незнакомец';
    meta.innerHTML='<span>'+fmtDate(ev.ts)+'</span><span class="count">'+total+'\u00a0чел.</span>';
  });
}
function poll(){
  fetch('detections.json?t='+Date.now()).then(r=>r.json()).then(evs=>{
    dot.className='live';
    const newTs=evs.length?evs[evs.length-1].ts:0;
    if(newTs!==lastTs){lastTs=newTs;rebuild(evs);}
  }).catch(()=>dot.className='');
}
poll();setInterval(poll,1000);
const cam=document.getElementById('cam');
setInterval(()=>{cam.src='frame.jpg?t='+Date.now();},334);
cam.src='frame.jpg?t=0';
</script></body></html>
"""


# ── Веб-сервер (in-memory) ──────────────────────────────────────────────────
class _WebHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?")[0]
        web = self.server.web
        if path in ("/", "/index.html"):
            self._send(_HTML.encode(), "text/html; charset=utf-8")
        elif path == "/frame.jpg":
            buf = web.get_frame_jpeg()
            if buf:
                self._send(buf, "image/jpeg")
            else:
                self.send_error(503)
        elif path == "/detections.json":
            data = json.dumps(
                web.event_store.recent(), ensure_ascii=False
            ).encode()
            self._send(data, "application/json")
        elif path.startswith("/snap/") and path.endswith(".jpg"):
            try:
                eid = int(path[6:-4])
            except ValueError:
                self.send_error(404)
                return
            jpeg = web.event_store.get_snapshot(eid)
            if jpeg:
                self._send(jpeg, "image/jpeg")
            else:
                self.send_error(404)
        elif path == "/health":
            self._send(
                json.dumps(web.get_health()).encode(), "application/json"
            )
        elif path == "/reload":
            web.reload_requested.set()
            self._send(b'{"ok":true}', "application/json")
        elif path == "/debug/aligned.jpg":
            buf = web._aligned_jpeg
            if buf:
                self._send(buf, "image/jpeg")
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def _send(self, data: bytes, content_type: str):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(data))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *_args):
        pass


class WebServer:
    def __init__(self, port: int, event_store: EventStore):
        self.event_store = event_store
        self.reload_requested = threading.Event()
        self._frame_jpeg: bytes | None = None
        self._frame_lock = threading.Lock()
        self._live_frame = None
        self._live_lock = threading.Lock()
        self._freeze_frame = None
        self._freeze_lock = threading.Lock()
        self._use_freeze = False
        self._aligned_jpeg: bytes | None = None
        self._start_time = time.time()
        self._last_detect_ts = 0.0

        httpd = http.server.ThreadingHTTPServer(("", port), _WebHandler)
        httpd.web = self
        self._httpd = httpd
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        threading.Thread(target=self._frame_writer, daemon=True).start()
        print(f"[*] Веб: http://0.0.0.0:{port}")

    def update_source(self, frame, has_known, has_stranger, stranger_conf):
        snap = frame.copy()
        if has_known or not has_stranger:
            # знакомый появился ИЛИ лиц нет → живой стрим
            self._use_freeze = False
        elif has_stranger:
            with self._freeze_lock:
                self._freeze_frame = snap
            if stranger_conf:
                self._use_freeze = True
        with self._live_lock:
            self._live_frame = snap

    def update_aligned(self, jpeg_bytes: bytes):
        self._aligned_jpeg = jpeg_bytes

    def notify_detection(self):
        self._last_detect_ts = time.time()

    def get_frame_jpeg(self) -> bytes | None:
        with self._frame_lock:
            return self._frame_jpeg

    def get_health(self) -> dict:
        return {
            "uptime_s": round(time.time() - self._start_time),
            "last_detection_ts": self._last_detect_ts,
            "frame_jpeg_bytes": len(self._frame_jpeg) if self._frame_jpeg else 0,
        }

    def shutdown(self):
        self._httpd.shutdown()

    def _frame_writer(self):
        while True:
            time.sleep(FRAME_INTERVAL)
            if self._use_freeze:
                with self._freeze_lock:
                    frame = self._freeze_frame
            else:
                with self._live_lock:
                    frame = self._live_frame
            if frame is None:
                continue
            try:
                ok, buf = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
                )
                if ok:
                    with self._frame_lock:
                        self._frame_jpeg = buf.tobytes()
            except Exception:
                pass


# ── Утилиты ──────────────────────────────────────────────────────────────────
def letterbox(img, size=None, fill=(114, 114, 114)):
    """Масштабирует с сохранением пропорций и добивает до квадрата."""
    if size is None:
        size = _SCRFD_INPUT
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    pad_h, pad_w = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), fill, dtype=np.uint8)
    out[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized
    return out, scale, (pad_w, pad_h)


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


def _draw_faces(frame, detected):
    """Рисует рамки и Unicode-подписи через PIL (только ROI полоски)."""
    if not detected:
        return
    h_frame, w_frame = frame.shape[:2]
    for bbox, name, score in detected:
        x1, y1, x2, y2 = map(int, bbox)
        color = (0, 200, 0) if name != UNKNOWN_LABEL else (0, 0, 200)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        lx1, ly1 = max(x1, 0), max(y2 - 30, 0)
        lx2, ly2 = min(x2, w_frame), min(y2, h_frame)
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


# ── SCRFD постобработка ─────────────────────────────────────────────────────
def _is_frontal(kps) -> tuple:
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_dist = abs(right_eye[0] - left_eye[0])
    if eye_dist < 1:
        return False, 1.0
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    offset = abs(nose[0] - eye_center_x) / eye_dist
    return offset < _FRONTAL_THRESH, offset


def _decode_scrfd(outputs, scale, pad):
    pad_w, pad_h = pad
    all_boxes, all_scores, all_kps = [], [], []
    n = len(_STRIDES)
    for i, stride in enumerate(_STRIDES):
        scores_raw = outputs[i].flatten()
        bboxes = outputs[i + n].reshape(-1, 4)
        kps = outputs[i + n * 2].reshape(-1, 10)
        ac = _ANCHORS[stride]
        if scores_raw.max() > 1.0 or scores_raw.min() < 0.0:
            scores = 1.0 / (1.0 + np.exp(-scores_raw))
        else:
            scores = scores_raw
        x1 = ac[:, 0] - bboxes[:, 0] * stride
        y1 = ac[:, 1] - bboxes[:, 1] * stride
        x2 = ac[:, 0] + bboxes[:, 2] * stride
        y2 = ac[:, 1] + bboxes[:, 3] * stride
        kp_x = ac[:, 0:1] + kps[:, 0::2] * stride
        kp_y = ac[:, 1:2] + kps[:, 1::2] * stride
        kps_dec = np.stack([kp_x, kp_y], axis=-1)
        mask = scores > _SCORE_THRESH
        if not mask.any():
            continue
        all_boxes.append(
            np.stack([x1[mask], y1[mask], x2[mask], y2[mask]], axis=1)
        )
        all_scores.append(scores[mask])
        all_kps.append(kps_dec[mask])
    if not all_boxes:
        return []
    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    kps_all = np.concatenate(all_kps)
    xywh = np.column_stack([
        boxes[:, 0], boxes[:, 1],
        boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1],
    ]).tolist()
    idxs = cv2.dnn.NMSBoxes(xywh, scores.tolist(), _SCORE_THRESH, _NMS_THRESH)
    if len(idxs) == 0:
        return []
    result = []
    for idx in idxs.flatten():
        box = boxes[idx].copy()
        kp = kps_all[idx].copy()
        box[[0, 2]] = (box[[0, 2]] - pad_w) / scale
        box[[1, 3]] = (box[[1, 3]] - pad_h) / scale
        kp[:, 0] = (kp[:, 0] - pad_w) / scale
        kp[:, 1] = (kp[:, 1] - pad_h) / scale
        result.append((box, kp))
    return result


# ── RKNN обёртки ─────────────────────────────────────────────────────────────
class _RKNNModel:
    def __init__(self, path, core_mask=None):
        self.net = RKNNLite()
        if self.net.load_rknn(path) != 0:
            raise RuntimeError(f"Не удалось загрузить модель: {path}")
        kwargs = {} if core_mask is None else {"core_mask": core_mask}
        if self.net.init_runtime(**kwargs) != 0:
            raise RuntimeError("Ошибка init_runtime RKNN")

    def _run(self, inputs):
        return self.net.inference(inputs=inputs)

    def release(self):
        self.net.release()


class FaceDetector(_RKNNModel):
    def __init__(self, path, core_mask=None):
        global _SCRFD_INPUT, _ANCHORS
        super().__init__(path, core_mask=core_mask)
        for candidate in [640, 480, 360, 320]:
            probe = np.zeros((1, candidate, candidate, 3), dtype=np.uint8)
            outs = self._run([probe])
            if outs is not None:
                if candidate != _SCRFD_INPUT:
                    print(f"[*] SCRFD: {candidate}×{candidate}")
                    _SCRFD_INPUT = candidate
                    _ANCHORS = _build_anchor_centers(candidate)
                break
        else:
            raise RuntimeError("Не удалось определить размер SCRFD")

    def detect(self, frame):
        img, scale, pad = letterbox(frame)
        inp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return _decode_scrfd(self._run([inp[np.newaxis]]), scale, pad)


class FaceEncoder(_RKNNModel):
    def encode(self, frame, kps):
        M, _ = cv2.estimateAffinePartial2D(
            kps, _ARCFACE_TPL, method=cv2.LMEDS
        )
        if M is None:
            return None, None
        aligned = cv2.warpAffine(frame, M, (112, 112), borderValue=0.0)
        inp = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        emb = self._run([inp[np.newaxis]])[0].flatten()
        norm = np.linalg.norm(emb)
        enc = emb / norm if norm > 0 else None
        return enc, aligned


# ── База лиц (с hot-reload) ─────────────────────────────────────────────────
def _load_faces_from_dir(directory, detector, encoder):
    encodings, names = [], []
    faces_dir = Path(directory)
    if not faces_dir.is_dir():
        print(f"[!] Папка {directory} не найдена.")
        return np.empty((0, 512), dtype=np.float32), []
    for person_dir in sorted(faces_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        loaded = 0
        for photo in person_dir.iterdir():
            if photo.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            img = cv2.imread(str(photo))
            if img is None:
                continue
            dets = detector.detect(img)
            if not dets:
                print(f"[!] Лицо не найдено: {photo}")
                continue
            bbox, kps = max(
                dets, key=lambda d: (d[0][2] - d[0][0]) * (d[0][3] - d[0][1])
            )
            enc, _ = encoder.encode(img, kps)
            if enc is None:
                continue
            encodings.append(enc)
            names.append(person_dir.name)
            loaded += 1
        if loaded:
            print(f"[+] {person_dir.name}: {loaded} фото")
    mat = (
        np.array(encodings, dtype=np.float32)
        if encodings
        else np.empty((0, 512), dtype=np.float32)
    )
    return mat, names


class FaceDB:
    def __init__(self, directory, detector, encoder):
        self.directory = directory
        self._detector = detector
        self._encoder = encoder
        self._lock = threading.Lock()
        self._encodings = np.empty((0, 512), dtype=np.float32)
        self._names: list = []
        self._name_index: dict = {}
        self._last_mtime = 0.0
        self._reload_flag = False
        self.reload()

    @property
    def encodings(self):
        with self._lock:
            return self._encodings

    @property
    def name_index(self):
        with self._lock:
            return self._name_index

    def request_reload(self):
        self._reload_flag = True

    def needs_reload(self) -> bool:
        if self._reload_flag:
            return True
        return self._dir_mtime() > self._last_mtime

    def reload(self):
        print("[*] Загрузка базы лиц...")
        encs, names = _load_faces_from_dir(
            self.directory, self._detector, self._encoder
        )
        idx = {}
        for i, nm in enumerate(names):
            idx.setdefault(nm, []).append(i)
        with self._lock:
            self._encodings = encs
            self._names = names
            self._name_index = idx
        self._last_mtime = self._dir_mtime()
        self._reload_flag = False
        print(f"[*] База: {len(set(names))} чел., {len(names)} фото")

    def _dir_mtime(self) -> float:
        root = Path(self.directory)
        if not root.is_dir():
            return 0.0
        mtime = root.stat().st_mtime
        for d in root.iterdir():
            if d.is_dir():
                try:
                    mtime = max(mtime, d.stat().st_mtime)
                except OSError:
                    pass
        return mtime


# ── Распознавание ────────────────────────────────────────────────────────────
def identify_face(encoding, known_encodings, name_index):
    if len(known_encodings) == 0:
        return []
    sims = known_encodings @ encoding
    top = []
    for n, idxs in name_index.items():
        best_n = float(max(sims[i] for i in idxs))
        top.append((n, best_n))
    top.sort(key=lambda x: -x[1])
    return top


# ── FaceTracker (стейт-машина) ───────────────────────────────────────────────
class FaceTracker:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.detected: list = []
        self.face_has_stranger = False
        self.face_stranger_conf = False
        self.face_has_known = False
        self.last_greeted: dict = {}
        self._confirm_streak: dict = {}
        self._stranger_streak = 0
        self._last_stranger_ts = 0.0
        self._last_event_time: dict = {}
        self._last_state: tuple = (frozenset(), 0)
        self._last_heartbeat = 0.0
        self._pending_stranger = None
        self._frame_count = 0

    def update(self, face_results, frame) -> list:
        events = []
        self._frame_count += 1

        if face_results is not None:
            self.detected = face_results
            current_names: set = set()
            raw_unk = 0

            for bbox, name, score in face_results:
                if name == UNKNOWN_LABEL:
                    raw_unk += 1
                    continue
                current_names.add(name)
                self._confirm_streak[name] = (
                    self._confirm_streak.get(name, 0) + 1
                )
                if self.cfg.debug:
                    print(
                        f"[D]   streak={self._confirm_streak[name]}"
                        f"/{self.cfg.confirm_frames}"
                    )
                if self._confirm_streak[name] >= self.cfg.confirm_frames:
                    now = time.time()
                    if now - self.last_greeted.get(name, 0) > self.cfg.greet_cooldown:
                        self.last_greeted[name] = now
                        events.append(("greet", name))

            # stranger streak (rate-limited)
            now_ts = time.time()
            if raw_unk > 0:
                if self._stranger_streak == 0 or (
                    now_ts - self._last_stranger_ts
                ) >= self.cfg.streak_min_interval:
                    self._stranger_streak += 1
                    self._last_stranger_ts = now_ts
                if self.cfg.debug:
                    print(
                        f"[D]   stranger_streak="
                        f"{self._stranger_streak}/{self.cfg.confirm_frames}"
                    )
            else:
                self._stranger_streak = 0
            unk_count = (
                raw_unk
                if self._stranger_streak >= self.cfg.confirm_frames
                else 0
            )

            self.face_has_stranger = raw_unk > 0
            self.face_stranger_conf = unk_count > 0
            self.face_has_known = bool(current_names)

            for gone in set(self._confirm_streak) - current_names - {UNKNOWN_LABEL}:
                self._confirm_streak[gone] = 0

            cur_state = (frozenset(current_names), unk_count)
            if cur_state != self._last_state:
                web_names = list(current_names) + [UNKNOWN_LABEL] * unk_count
                if web_names:
                    now = time.time()
                    key = cur_state
                    if current_names:
                        self._pending_stranger = None
                        if now - self._last_event_time.get(key, 0) >= self.cfg.web_event_cooldown:
                            self._last_event_time[key] = now
                            events.append((
                                "web_event", web_names,
                                frame.copy(), list(self.detected),
                            ))
                    else:
                        if self._pending_stranger is None:
                            self._pending_stranger = (
                                frame.copy(), list(self.detected),
                                now, web_names, key,
                            )
                            if self.cfg.debug:
                                print(
                                    f"[D] {time.strftime('%H:%M:%S')} "
                                    f"незнакомец в ожидании "
                                    f"({self.cfg.stranger_confirm_delay:.0f}с)..."
                                )
                else:
                    self._pending_stranger = None
                    if self.cfg.debug:
                        print(
                            f"[D] {time.strftime('%H:%M:%S')} "
                            f"кадр {self._frame_count}: лиц нет"
                        )
                self._last_state = cur_state
                self._last_heartbeat = time.time()
            elif (
                not current_names
                and raw_unk > 0
                and self._pending_stranger is None
            ):
                now = time.time()
                pn = [UNKNOWN_LABEL] * raw_unk
                pk = (frozenset(), raw_unk)
                self._pending_stranger = (
                    frame.copy(), list(self.detected), now, pn, pk,
                )
                if self.cfg.debug:
                    print(
                        f"[D] {time.strftime('%H:%M:%S')} "
                        f"незнакомец (pre-streak) в ожидании..."
                    )
            elif (
                self.cfg.debug
                and not current_names
                and not unk_count
                and time.time() - self._last_heartbeat >= DEBUG_INTERVAL
            ):
                print(
                    f"[D] {time.strftime('%H:%M:%S')} "
                    f"кадр {self._frame_count}: лиц нет"
                )
                self._last_heartbeat = time.time()

        # stranger timer (каждый кадр)
        if self._pending_stranger is not None:
            sf, sd, st, sn, sk = self._pending_stranger
            if time.time() - st >= self.cfg.stranger_confirm_delay:
                self._pending_stranger = None
                now = time.time()
                if now - self._last_event_time.get(sk, 0) >= self.cfg.web_event_cooldown:
                    self._last_event_time[sk] = now
                    events.append(("web_event", sn, sf, sd))
                if self.cfg.debug:
                    print(
                        f"[D] {time.strftime('%H:%M:%S')} "
                        f"незнакомец подтверждён → событие"
                    )

        return events


# ── Диспетчер событий ────────────────────────────────────────────────────────
def _dispatch(ev, cfg, event_store, web):
    if ev[0] == "greet" and not cfg.no_tts:
        greeting = GREETINGS[cfg.lang].format(ev[1])
        print(f"[>] {greeting}")
        speak(greeting, cfg.lang)
    elif ev[0] == "web_event":
        _, names, snap, det = ev
        _draw_faces(snap, det)
        ok, buf = cv2.imencode(".jpg", snap, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            event_store.add(names, buf.tobytes())
        if web:
            web.notify_detection()


# ── Камера ───────────────────────────────────────────────────────────────────
def _open_camera(cfg: Config):
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
                print(f"[*] Камера idx={idx}: {w}×{h}")
                return c, idx
        c.release()
    return None, -1


# ── Главный цикл ────────────────────────────────────────────────────────────
def main():
    cfg = _parse_args()
    os.makedirs(cfg.data_dir, exist_ok=True)

    event_store = EventStore(cfg.db_path)
    web = WebServer(cfg.web_port, event_store) if cfg.web_port else None

    print("[*] Инициализация NPU...")
    detector = FaceDetector(SCRFD_MODEL, core_mask=1)
    encoder = FaceEncoder(ARCFACE_MODEL, core_mask=2)
    face_db = FaceDB(cfg.known_faces_dir, detector, encoder)
    tracker = FaceTracker(cfg)

    # восстанавливаем last_greeted из БД (не здороваемся повторно после рестарта)
    tracker.last_greeted.update(
        event_store.last_greeted_times(cfg.greet_cooldown)
    )

    cap, _ = _open_camera(cfg)
    if cap is None:
        print("[!] Камера не найдена")
        detector.release()
        encoder.release()
        return

    # ── async encode pipeline (Core1) ────────────────────────────────────
    _enc_in: queue.Queue = queue.Queue(maxsize=2)
    _enc_out: queue.Queue = queue.Queue()  # без лимита — защита от deadlock

    def _encode_loop():
        score_hist: dict = {}
        while True:
            item = _enc_in.get()
            if item is None:
                break
            if item == "pause":
                _enc_out.put("paused")
                _enc_in.get()  # ждём "resume"
                score_hist.clear()
                continue

            frm, raw_faces = item
            encodings = face_db.encodings
            name_idx = face_db.name_index
            results = []

            for bbox, kps in raw_faces:
                frontal, offset = _is_frontal(kps)
                if not frontal:
                    if cfg.debug:
                        print(
                            f"[D] {time.strftime('%H:%M:%S')} "
                            f"профиль offset={offset:.3f}"
                        )
                    continue
                enc, aligned = encoder.encode(frm, kps)
                if enc is None:
                    continue

                top = identify_face(enc, encodings, name_idx)
                if top:
                    best_cand, best_raw = top[0]
                    hist = score_hist.setdefault(
                        best_cand, deque(maxlen=cfg.score_window)
                    )
                    hist.append(best_raw)
                    avg = float(np.mean(hist))
                    name = (
                        best_cand
                        if avg >= cfg.recognition_threshold
                        else UNKNOWN_LABEL
                    )
                    score = avg
                    if name == UNKNOWN_LABEL and score < cfg.stranger_min_score:
                        continue
                else:
                    name, score = UNKNOWN_LABEL, 0.0

                if cfg.debug:
                    cands = "  ".join(f"{n}={s:.3f}" for n, s in top)
                    st = "✓" if name != UNKNOWN_LABEL else "?"
                    nh = len(score_hist.get(top[0][0], [])) if top else 0
                    print(
                        f"[D] {time.strftime('%H:%M:%S')} {st} "
                        f"best={name} avg{nh}={score:.3f}  "
                        f"off={offset:.3f}  [{cands}]"
                    )
                    if aligned is not None and web:
                        try:
                            ok, buf = cv2.imencode(".jpg", aligned)
                            if ok:
                                web.update_aligned(buf.tobytes())
                        except Exception:
                            pass

                results.append((bbox, name, score))
            _enc_out.put(results)

    enc_thread = threading.Thread(target=_encode_loop, daemon=True)
    enc_thread.start()

    # ── signal handling ──────────────────────────────────────────────────
    _running = True

    def _on_signal(sig, _):
        nonlocal _running
        _running = False
        print(f"\n[*] Сигнал {sig}, завершаем...")

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    print("[*] Запуск. Q — выход.")
    frame_count = 0
    no_frame_count = 0
    last_reload_check = time.time()

    # ── main loop ────────────────────────────────────────────────────────
    while _running:
        ret, frame = cap.read()
        if not ret:
            no_frame_count += 1
            if no_frame_count >= cfg.no_frame_reconnect_after:
                print("[*] Переподключение камеры...")
                cap.release()
                time.sleep(cfg.no_frame_reconnect_delay)
                cap, _ = _open_camera(cfg)
                if cap is None:
                    print("[!] Камера не найдена")
                    break
                no_frame_count = 0
            time.sleep(0.1)
            continue

        no_frame_count = 0
        frame_count += 1

        # ── hot-reload check ─────────────────────────────────────────
        now = time.time()
        check_reload = now - last_reload_check >= RELOAD_CHECK_S
        if web and web.reload_requested.is_set():
            web.reload_requested.clear()
            face_db.request_reload()
            check_reload = True
        if check_reload:
            last_reload_check = now
            if face_db.needs_reload():
                # drain pending results
                while True:
                    try:
                        msg = _enc_out.get_nowait()
                        for ev in tracker.update(msg, frame):
                            _dispatch(ev, cfg, event_store, web)
                    except queue.Empty:
                        break
                _enc_in.put("pause")
                while True:
                    msg = _enc_out.get(timeout=10)
                    if msg == "paused":
                        break
                    for ev in tracker.update(msg, frame):
                        _dispatch(ev, cfg, event_store, web)
                face_db.reload()
                _enc_in.put("resume")

        # ── detect + encode ──────────────────────────────────────────
        if frame_count % cfg.process_every_n == 0:
            faces = detector.detect(frame)
            try:
                _enc_in.put_nowait((frame.copy(), faces))
            except queue.Full:
                pass

        try:
            face_results = _enc_out.get_nowait()
        except queue.Empty:
            face_results = None

        # ── track + dispatch ─────────────────────────────────────────
        for ev in tracker.update(face_results, frame):
            _dispatch(ev, cfg, event_store, web)

        _draw_faces(frame, tracker.detected)

        if web:
            web.update_source(
                frame,
                tracker.face_has_known,
                tracker.face_has_stranger,
                tracker.face_stranger_conf,
            )

        if cfg.show_display:
            cv2.imshow("Faces", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # ── cleanup ──────────────────────────────────────────────────────
    _enc_in.put(None)
    enc_thread.join(timeout=2)
    if cap is not None:
        cap.release()
    if cfg.show_display:
        cv2.destroyAllWindows()
    detector.release()
    encoder.release()
    if web:
        web.shutdown()
    event_store.close()
    print("[*] Завершено.")


if __name__ == "__main__":
    main()
