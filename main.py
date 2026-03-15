#!/usr/bin/env python3
"""
Распознавание лиц на NPU (RKNN) — Orange Pi 5 Max / RK3588.
Детекция: SCRFD-2.5G  |  Энкодинг: MobileFaceNet/ArcFace
Скачать модели: bash download_models.sh
"""

import json
import math
import os
import sys
import time
import subprocess
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from rknnlite.api import RKNNLite

# ── Пути к моделям ──────────────────────────────────────────────────────────
SCRFD_MODEL   = "models/scrfd.rknn"
ARCFACE_MODEL = "models/arcface.rknn"

# ── Настройки ────────────────────────────────────────────────────────────────
KNOWN_FACES_DIR       = "known_faces"
CAMERA_INDEX          = 0
GREET_COOLDOWN        = 10    # секунд между приветствиями одного человека
PROCESS_EVERY_N       = 1     # обрабатывать каждый N-й кадр
CONFIRM_FRAMES        = 3     # сколько кадров подряд нужно видеть человека перед приветствием
SHOW_DISPLAY          = False # показывать окно (требует X-сервер/дисплей)
DEBUG                 = True  # подробный вывод в консоль
DEBUG_INTERVAL        = 5.0   # секунд между пульсом (если нет детекций)
WEB_PORT              = 8080  # порт веб-интерфейса (0 = выключен)
WEB_DIR               = "/tmp/faces_web"
FRAME_FILE            = WEB_DIR + "/frame.jpg"
DETECTIONS_FILE       = WEB_DIR + "/detections.json"
SNAPSHOTS_DIR         = WEB_DIR + "/snap"
SNAPSHOTS_MAX         = 10
FRAME_INTERVAL        = 1.0
WEB_EVENT_COOLDOWN    = 60.0  # секунд: не повторять событие для тех же людей

# Порог распознавания — косинусное сходство (0..1, выше = строже)
RECOGNITION_THRESHOLD = 0.55

GREETINGS     = {"ru": "Привет, {}!", "en": "Hello, {}!"}
LANG          = "ru"
UNKNOWN_LABEL = "Незнакомец"
ESPEAK_ARGS   = {"ru": ["-v", "ru", "-s", "140"], "en": ["-v", "en"]}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# ── SCRFD внутренние параметры ───────────────────────────────────────────────
_SCRFD_INPUT  = 360   # модель скомпилирована под 360×360
_STRIDES      = [8, 16, 32]
_NUM_ANCHORS  = 2
_SCORE_THRESH  = 0.50
_NMS_THRESH    = 0.40
_FRONTAL_THRESH = 0.20  # макс. смещение носа от центра глаз (доля расстояния между глазами)

# Шаблон ключевых точек лица для выравнивания 112×112 (стандарт ArcFace/InsightFace)
_ARCFACE_TPL = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


# ── Якорные центры (вычисляются один раз при старте) ────────────────────────
def _build_anchor_centers():
    centers = {}
    for stride in _STRIDES:
        fs = math.ceil(_SCRFD_INPUT / stride)
        gy, gx = np.mgrid[0:fs, 0:fs]
        c = np.stack([gx * stride, gy * stride], axis=-1).reshape(-1, 2)
        # Каждую точку сетки повторяем _NUM_ANCHORS раз
        centers[stride] = np.repeat(c, _NUM_ANCHORS, axis=0).astype(np.float32)
    return centers

_ANCHORS = _build_anchor_centers()


# ── Веб: данные ──────────────────────────────────────────────────────────────
_detection_log: deque  = deque(maxlen=SNAPSHOTS_MAX)
_snapshot_counter: int = 0
_latest_frame_bgr      = None
_latest_frame_lock     = threading.Lock()

_HTML = """<!DOCTYPE html>
<html lang="ru"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Камера</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{background:#0f0f0f;color:#e0e0e0;font-family:-apple-system,BlinkMacSystemFont,sans-serif;
  display:flex;flex-direction:column;height:100%;
  padding:12px;padding-top:max(12px,env(safe-area-inset-top));gap:10px}
header{display:flex;align-items:center;gap:8px;flex-shrink:0}
#dot{width:8px;height:8px;border-radius:50%;background:#555;transition:background .3s}
#dot.live{background:#30d158}
header span{font-size:.8rem;font-weight:600;color:#555;letter-spacing:.08em;text-transform:uppercase}
#main{display:flex;gap:10px;flex:1;min-height:0}
#cam-wrap{background:#1c1c1e;border-radius:12px;overflow:hidden;flex-shrink:0;width:240px;
  display:flex;align-items:center;justify-content:center}
#cam{width:100%;height:100%;object-fit:contain;display:block}
#feed-wrap{flex:1;overflow-y:auto;display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;align-content:start}
.card{background:#1c1c1e;border-radius:10px;overflow:hidden;display:flex;flex-direction:column}
.thumb{width:100%;aspect-ratio:4/3;object-fit:cover;display:block;background:#111}
.info{padding:6px 8px}
.names{font-size:.8rem;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.names.k{color:#30d158}.names.u{color:#ff453a}
.ts{font-size:.65rem;color:#555;margin-top:2px}
#empty{color:#333;font-size:.85rem;margin:auto;text-align:center}
</style></head><body>
<header><span id="dot"></span><span>Камера</span></header>
<div id="main">
  <div id="cam-wrap"><img id="cam" alt=""></div>
  <div id="feed-wrap"><p id="empty">Ожидание...</p></div>
</div>
<script>
const feed=document.getElementById('feed-wrap'),dot=document.getElementById('dot');
let lastTs=0;
function fmt(ts){
  const s=(Date.now()-ts*1000)/1000;
  if(s<60)return'только что';
  if(s<3600)return Math.floor(s/60)+'\u00a0мин назад';
  return new Date(ts*1000).toLocaleTimeString('ru-RU',{hour:'2-digit',minute:'2-digit'});
}
function rebuild(evs){
  feed.innerHTML='';
  if(!evs.length){feed.innerHTML='<p id="empty">Ожидание...</p>';return;}
  evs.slice().reverse().forEach(ev=>{
    const kn=ev.names.filter(n=>n!=='Незнакомец');
    const un=ev.names.filter(n=>n==='Незнакомец').length;
    const hasKnown=kn.length>0;
    let label=kn.join(', ');
    if(un>0)label+=(label?'\u00a0+\u00a0':'')+(un===1?'Незнакомец':un+'\u00a0незн.');
    const d=document.createElement('div');
    d.className='card';d.dataset.ts=ev.ts;
    d.innerHTML=`<img class="thumb" src="snap/${ev.img}.jpg?t=${ev.ts}" loading="lazy">
    <div class="info"><div class="names ${hasKnown?'k':'u'}">${label}</div><div class="ts">${fmt(ev.ts)}</div></div>`;
    feed.appendChild(d);
  });
}
function poll(){
  fetch('detections.json?t='+Date.now()).then(r=>r.json()).then(evs=>{
    dot.className='live';
    const newTs=evs.length?evs[evs.length-1].ts:0;
    if(newTs!==lastTs){lastTs=newTs;rebuild(evs);}
  }).catch(()=>dot.className='');
}
setInterval(()=>feed.querySelectorAll('.card').forEach(c=>{
  c.querySelector('.ts').textContent=fmt(+c.dataset.ts);}),15000);
poll();setInterval(poll,1000);
const cam=document.getElementById('cam');
setInterval(()=>{cam.src='frame.jpg?t='+Date.now();},1000);
cam.src='frame.jpg?t=0';
</script></body></html>
"""


def _write_file(path: str, data: bytes):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def _web_add_event(names: list, frame):
    global _snapshot_counter
    slot = _snapshot_counter % SNAPSHOTS_MAX
    _snapshot_counter += 1

    try:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            _write_file(os.path.join(SNAPSHOTS_DIR, f"{slot}.jpg"), buf.tobytes())
    except OSError:
        pass

    ev = {"ts": round(time.time(), 2), "names": sorted(names), "img": slot}
    _detection_log.append(ev)

    try:
        _write_file(DETECTIONS_FILE,
                    json.dumps(list(_detection_log), ensure_ascii=False).encode())
    except OSError:
        pass


def _frame_writer():
    """Фоновый поток: раз в FRAME_INTERVAL секунд сохраняет текущий кадр в файл."""
    while True:
        time.sleep(FRAME_INTERVAL)
        with _latest_frame_lock:
            frame = _latest_frame_bgr
        if frame is None:
            continue
        try:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                _write_file(FRAME_FILE, buf.tobytes())
        except OSError:
            pass


def _start_web_server():
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    _write_file(WEB_DIR + "/index.html", _HTML.encode())
    _write_file(DETECTIONS_FILE, b"[]")
    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(WEB_PORT), "--directory", WEB_DIR],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"[*] Веб-интерфейс: http://0.0.0.0:{WEB_PORT}  (pid {proc.pid})")
    return proc


# ── Утилиты ──────────────────────────────────────────────────────────────────
def letterbox(img, size=_SCRFD_INPUT, fill=(114, 114, 114)):
    """Масштабирует с сохранением пропорций и добивает до квадрата."""
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    pad_h, pad_w = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), fill, dtype=np.uint8)
    out[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized
    return out, scale, (pad_w, pad_h)


def speak(text: str):
    """Произносит текст через espeak-ng в отдельном потоке."""
    def _run():
        try:
            args = ["espeak-ng"] + ESPEAK_ARGS.get(LANG, []) + [text]
            subprocess.run(args, check=True, capture_output=True)
        except FileNotFoundError:
            print(f"[!] espeak-ng не установлен. Приветствие: {text}")
        except subprocess.CalledProcessError as e:
            print(f"[!] Ошибка TTS: {e}")
    threading.Thread(target=_run, daemon=True).start()


def draw_box(frame, x1, y1, x2, y2, name, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(frame, (x1, y2 - 28), (x2, y2), color, cv2.FILLED)
    cv2.putText(frame, name, (x1 + 6, y2 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)


# ── SCRFD постобработка ──────────────────────────────────────────────────────
def _is_frontal(kps) -> tuple:
    """Возвращает (is_frontal: bool, offset: float).
    kps: (5, 2) — [left_eye, right_eye, nose, left_mouth, right_mouth]
    Проверяем, что нос находится симметрично между глазами по горизонтали.
    """
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_dist = abs(right_eye[0] - left_eye[0])
    if eye_dist < 1:
        return False, 1.0
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    offset = abs(nose[0] - eye_center_x) / eye_dist
    return offset < _FRONTAL_THRESH, offset


def _decode_scrfd(outputs, scale, pad):
    """
    Декодирует 9 выходных тензоров SCRFD → список (bbox[4 float], kps[5,2]).

    Ожидаемый порядок тензоров (rknn_model_zoo SCRFD-2.5G, stride 8→16→32):
      outputs[i*3+0] — score  (N,)    — уже после sigmoid, значения 0..1
      outputs[i*3+1] — bbox   (N, 4)  — lt/rb дистанции в единицах stride
      outputs[i*3+2] — kps    (N, 10) — смещения xy для 5 точек в единицах stride
    Если модель возвращает другой порядок — поправь _STRIDES или переставь индексы.
    """
    pad_w, pad_h = pad
    all_boxes, all_scores, all_kps = [], [], []

    n = len(_STRIDES)
    for i, stride in enumerate(_STRIDES):
        scores_raw = outputs[i + 0].flatten()
        bboxes = outputs[i + n].reshape(-1, 4)
        kps    = outputs[i + n * 2].reshape(-1, 10)
        ac     = _ANCHORS[stride]

        # Применяем sigmoid если скоры — логиты (не в диапазоне 0..1)
        if scores_raw.max() > 1.0 or scores_raw.min() < 0.0:
            scores = 1.0 / (1.0 + np.exp(-scores_raw))
        else:
            scores = scores_raw

        # Декодирование bbox: дистанции (left,top,right,bottom) от центра якоря
        x1 = ac[:, 0] - bboxes[:, 0] * stride
        y1 = ac[:, 1] - bboxes[:, 1] * stride
        x2 = ac[:, 0] + bboxes[:, 2] * stride
        y2 = ac[:, 1] + bboxes[:, 3] * stride

        # Декодирование ключевых точек: смещения от центра якоря
        kp_x = ac[:, 0:1] + kps[:, 0::2] * stride   # (N, 5)
        kp_y = ac[:, 1:2] + kps[:, 1::2] * stride   # (N, 5)
        kps_dec = np.stack([kp_x, kp_y], axis=-1)    # (N, 5, 2)

        mask = scores > _SCORE_THRESH
        if not mask.any():
            continue

        all_boxes.append(np.stack([x1[mask], y1[mask], x2[mask], y2[mask]], axis=1))
        all_scores.append(scores[mask])
        all_kps.append(kps_dec[mask])

    if not all_boxes:
        return []

    boxes   = np.concatenate(all_boxes)
    scores  = np.concatenate(all_scores)
    kps_all = np.concatenate(all_kps)

    # NMS
    xywh = np.column_stack([boxes[:, 0], boxes[:, 1],
                             boxes[:, 2] - boxes[:, 0],
                             boxes[:, 3] - boxes[:, 1]]).tolist()
    idxs = cv2.dnn.NMSBoxes(xywh, scores.tolist(), _SCORE_THRESH, _NMS_THRESH)
    if len(idxs) == 0:
        return []

    result = []
    for idx in idxs.flatten():
        box = boxes[idx].copy()
        kp  = kps_all[idx].copy()
        # Обратное масштабирование → координаты исходного кадра
        box[[0, 2]] = (box[[0, 2]] - pad_w) / scale
        box[[1, 3]] = (box[[1, 3]] - pad_h) / scale
        kp[:, 0]    = (kp[:, 0] - pad_w) / scale
        kp[:, 1]    = (kp[:, 1] - pad_h) / scale
        result.append((box, kp))

    return result


# ── Обёртки RKNN моделей ─────────────────────────────────────────────────────
class _RKNNModel:
    def __init__(self, path):
        self.net = RKNNLite()
        if self.net.load_rknn(path) != 0:
            raise RuntimeError(f"Не удалось загрузить модель: {path}")
        if self.net.init_runtime() != 0:
            raise RuntimeError("Ошибка инициализации RKNN runtime")

    def _run(self, inputs):
        return self.net.inference(inputs=inputs)

    def release(self):
        self.net.release()


class FaceDetector(_RKNNModel):
    def detect(self, frame):
        """Возвращает list[(bbox: ndarray[4], kps: ndarray[5,2])]."""
        img, scale, pad = letterbox(frame)
        inp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # uint8 RGB, NHWC
        return _decode_scrfd(self._run([inp[np.newaxis]]), scale, pad)


class FaceEncoder(_RKNNModel):
    def encode(self, frame, kps):
        """Выравнивает лицо по 5 точкам, возвращает (L2-норм. эмбеддинг (512,), aligned_img) или (None, None)."""
        M, _ = cv2.estimateAffinePartial2D(kps, _ARCFACE_TPL, method=cv2.LMEDS)
        if M is None:
            return None, None
        aligned = cv2.warpAffine(frame, M, (112, 112), borderValue=0.0)
        inp = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)   # uint8 RGB, NHWC
        emb = self._run([inp[np.newaxis]])[0].flatten()
        norm = np.linalg.norm(emb)
        enc = emb / norm if norm > 0 else None
        return enc, aligned


# ── База лиц ─────────────────────────────────────────────────────────────────
def load_known_faces(directory, detector: FaceDetector, encoder: FaceEncoder):
    encodings, names = [], []
    faces_dir = Path(directory)

    if not faces_dir.is_dir():
        print(f"[!] Папка {directory} не найдена. Создай её и добавь фото.")
        return np.empty((0, 512), dtype=np.float32), names

    for person_dir in sorted(faces_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        loaded = 0
        for photo_path in person_dir.iterdir():
            if photo_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            img = cv2.imread(str(photo_path))
            if img is None:
                print(f"[!] Не удалось прочитать: {photo_path}")
                continue
            detections = detector.detect(img)
            if not detections:
                print(f"[!] Лицо не найдено: {photo_path}")
                continue
            # Берём наибольшее лицо на фото
            bbox, kps = max(detections,
                            key=lambda d: (d[0][2] - d[0][0]) * (d[0][3] - d[0][1]))
            enc, _ = encoder.encode(img, kps)
            if enc is None:
                continue
            encodings.append(enc)
            names.append(person_dir.name)
            loaded += 1

        if loaded:
            print(f"[+] Загружено {loaded} фото: {person_dir.name}")
        else:
            print(f"[!] Нет подходящих фото в {person_dir}")

    print(f"[*] Всего в базе: {len(set(names))} человек(а)")
    return (np.array(encodings, dtype=np.float32) if encodings
            else np.empty((0, 512), dtype=np.float32)), names


# ── Распознавание ─────────────────────────────────────────────────────────────
def identify_face(encoding, known_encodings, known_names):
    if len(known_encodings) == 0:
        return UNKNOWN_LABEL, 0.0, []
    # Косинусное сходство (оба вектора уже L2-нормализованы → просто dot product)
    sims = known_encodings @ encoding
    best = int(np.argmax(sims))
    score = float(sims[best])
    name = known_names[best] if score >= RECOGNITION_THRESHOLD else UNKNOWN_LABEL
    # Топ кандидаты для дебага
    unique_names = sorted(set(known_names))
    top = []
    for n in unique_names:
        idxs = [i for i, nm in enumerate(known_names) if nm == n]
        best_n = float(max(sims[i] for i in idxs))
        top.append((n, best_n))
    top.sort(key=lambda x: -x[1])
    return name, score, top


# ── Главный цикл ─────────────────────────────────────────────────────────────
def main():
    global _latest_frame_bgr
    web_proc = None
    if WEB_PORT:
        web_proc = _start_web_server()
        threading.Thread(target=_frame_writer, daemon=True).start()

    print("[*] Инициализация NPU моделей...")
    detector = FaceDetector(SCRFD_MODEL)
    encoder  = FaceEncoder(ARCFACE_MODEL)

    print("[*] Загрузка базы лиц...")
    known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR, detector, encoder)

    print(f"[*] Открытие камеры (индекс {CAMERA_INDEX})...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[!] Не удаётся открыть камеру {CAMERA_INDEX}")
        detector.release()
        encoder.release()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("[*] Запуск. Нажми Q для выхода.")
    last_greeted: dict[str, float] = {}
    confirm_streak: dict[str, int] = {}  # сколько кадров подряд видим человека
    last_heartbeat = [0.0]
    last_debug_names: set[str] = set()
    last_event_time: dict[frozenset, float] = {}  # кулдаун веб-событий
    frame_count = 0
    detected: list[tuple] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Нет кадра с камеры")
            time.sleep(0.1)
            continue

        frame_count += 1

        if frame_count % PROCESS_EVERY_N == 0:
            detected = []
            faces = detector.detect(frame)

            current_names: set[str] = set()
            for bbox, kps in faces:
                frontal, frontal_offset = _is_frontal(kps)
                if not frontal:
                    if DEBUG:
                        ts = time.strftime("%H:%M:%S")
                        print(f"[D] {ts} пропуск: боковой профиль offset={frontal_offset:.3f} (thresh={_FRONTAL_THRESH})")
                    continue
                enc, aligned = encoder.encode(frame, kps)
                if enc is not None:
                    name, score, top_candidates = identify_face(enc, known_encodings, known_names)
                else:
                    name, score, top_candidates = UNKNOWN_LABEL, 0.0, []
                    aligned = None
                detected.append((bbox, name, score))
                current_names.add(name)

                if DEBUG:
                    ts = time.strftime("%H:%M:%S")
                    cands = "  ".join(f"{n}={s:.3f}" for n, s in top_candidates)
                    status = "✓" if name != UNKNOWN_LABEL else "?"
                    print(f"[D] {ts} {status} best={name} score={score:.3f}  offset={frontal_offset:.3f}  [{cands}]  thresh={RECOGNITION_THRESHOLD}")
                    # Сохраняем выровненное лицо для ручной проверки
                    if aligned is not None and WEB_PORT:
                        try:
                            debug_path = os.path.join(WEB_DIR, "aligned_last.jpg")
                            ok, buf = cv2.imencode(".jpg", aligned)
                            if ok:
                                _write_file(debug_path, buf.tobytes())
                        except OSError:
                            pass

                if name != UNKNOWN_LABEL:
                    confirm_streak[name] = confirm_streak.get(name, 0) + 1
                    if DEBUG:
                        print(f"[D]   streak={confirm_streak[name]}/{CONFIRM_FRAMES}")
                    if confirm_streak[name] >= CONFIRM_FRAMES:
                        now = time.time()
                        if now - last_greeted.get(name, 0) > GREET_COOLDOWN:
                            last_greeted[name] = now
                            greeting = GREETINGS[LANG].format(name)
                            print(f"[>] {greeting}")
                            speak(greeting)

            # Сбрасываем стрик для тех, кого нет в текущем кадре
            for gone in set(confirm_streak) - current_names - {UNKNOWN_LABEL}:
                confirm_streak[gone] = 0

            if current_names != last_debug_names:
                if current_names:
                    key = frozenset(current_names)
                    now = time.time()
                    if now - last_event_time.get(key, 0) >= WEB_EVENT_COOLDOWN:
                        last_event_time[key] = now
                        _web_add_event(list(current_names), frame)
                elif DEBUG:
                    ts = time.strftime("%H:%M:%S")
                    print(f"[D] {ts} кадр {frame_count}: лиц не обнаружено")
                last_debug_names = current_names
                last_heartbeat[0] = time.time()
            elif DEBUG and not current_names and time.time() - last_heartbeat[0] >= DEBUG_INTERVAL:
                ts = time.strftime("%H:%M:%S")
                print(f"[D] {ts} кадр {frame_count}: лиц не обнаружено")
                last_heartbeat[0] = time.time()

        for bbox, name, score in detected:
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 200, 0) if name != UNKNOWN_LABEL else (0, 0, 200)
            label = f"{name} {score:.2f}"
            draw_box(frame, x1, y1, x2, y2, label, color)

        if WEB_PORT:
            with _latest_frame_lock:
                _latest_frame_bgr = frame

        if SHOW_DISPLAY:
            cv2.imshow("Faces", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if SHOW_DISPLAY:
        cv2.destroyAllWindows()
    detector.release()
    encoder.release()
    if web_proc:
        web_proc.terminate()
    print("[*] Завершено.")


if __name__ == "__main__":
    main()
