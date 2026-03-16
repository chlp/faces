#!/usr/bin/env python3
"""
Распознавание лиц на NPU (RKNN) — Orange Pi 5 Max / RK3588.
Детекция: SCRFD-2.5G  |  Энкодинг: MobileFaceNet/ArcFace
Скачать модели: bash download_models.sh
"""

import json
import math
import os
import queue
import sys
import time
import subprocess
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rknnlite.api import RKNNLite

# ── Пути к моделям ──────────────────────────────────────────────────────────
SCRFD_MODEL   = "models/scrfd.rknn"
ARCFACE_MODEL = "models/arcface.rknn"

# ── Настройки ────────────────────────────────────────────────────────────────
KNOWN_FACES_DIR       = "known_faces"
CAMERA_MAX_INDEX      = 10    # перебирать индексы 0..N в поисках первой доступной камеры
NO_FRAME_RECONNECT_AFTER = 10 # после стольких подряд "нет кадра" — ждать 5 с и искать другую камеру
NO_FRAME_RECONNECT_DELAY  = 5 # секунд ждать перед переподключением
GREET_COOLDOWN        = 10    # секунд между приветствиями одного человека
PROCESS_EVERY_N       = 1     # обрабатывать каждый N-й кадр
CONFIRM_FRAMES        = 3     # сколько кадров подряд нужно видеть человека перед приветствием
STREAK_MIN_INTERVAL   = 1/3.0 # секунд между инкрементами стрика (3 стрика ≈ 1 с)
SCORE_WINDOW          = 7     # усреднение скора за последние N кадров (сглаживание)
SHOW_DISPLAY          = False # показывать окно (требует X-сервер/дисплей)
DEBUG                 = True  # подробный вывод в консоль
DEBUG_INTERVAL        = 5.0   # секунд между пульсом (если нет детекций)
WEB_PORT              = 8080  # порт веб-интерфейса (0 = выключен)
WEB_DIR               = "/tmp/faces_web"
FRAME_FILE            = WEB_DIR + "/frame.jpg"
DETECTIONS_FILE       = WEB_DIR + "/detections.json"
SNAPSHOTS_DIR         = WEB_DIR + "/snap"
SNAPSHOTS_MAX         = 15
FRAME_INTERVAL        = 0.1
WEB_EVENT_COOLDOWN    = 30.0  # секунд: не повторять событие для тех же людей
STRANGER_CONFIRM_DELAY = 5.0  # секунд ожидания перед записью незнакомца (вдруг опознают)

# Порог распознавания — косинусное сходство (0..1, выше = строже)
RECOGNITION_THRESHOLD = 0.45
# Минимальное сходство для учёта как «незнакомец»: ниже — не считаем лицом (отсекаем ложные детекции)
STRANGER_MIN_SCORE = 0.30

GREETINGS     = {"ru": "Привет, {}!", "en": "Hello, {}!"}
LANG          = "ru"
UNKNOWN_LABEL = "Незнакомец"
ESPEAK_ARGS   = {"ru": ["-v", "ru", "-s", "140"], "en": ["-v", "en"]}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# ── SCRFD внутренние параметры ───────────────────────────────────────────────
_SCRFD_INPUT  = 640   # будет переопределён автоматически после загрузки модели
_STRIDES      = [8, 16, 32]
_NUM_ANCHORS  = 2
_SCORE_THRESH  = 0.50
_NMS_THRESH    = 0.40
_FRONTAL_THRESH = 0.60  # макс. смещение носа от центра глаз (доля расстояния между глазами)

# Шаблон ключевых точек лица для выравнивания 112×112 (стандарт ArcFace/InsightFace)
_ARCFACE_TPL = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)


# ── Шрифт для Unicode-подписей на видео ─────────────────────────────────────
def _load_ui_font(size: int = 20) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"[*] Шрифт: {path}")
            return ImageFont.truetype(path, size)
    print("[!] TTF-шрифт не найден, кириллица может не отображаться")
    return ImageFont.load_default()

_UI_FONT = _load_ui_font()


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
html,body{height:100%;background:#0a0a0a;color:#e0e0e0;
  font-family:-apple-system,BlinkMacSystemFont,sans-serif}
body{display:grid;grid-template-rows:auto 1fr;
  padding:10px;padding-top:max(10px,env(safe-area-inset-top));gap:10px;overflow:hidden}
header{display:flex;align-items:center;gap:8px}
#dot{width:8px;height:8px;border-radius:50%;background:#555;transition:background .3s;flex-shrink:0}
#dot.live{background:#30d158}
header span{font-size:.75rem;font-weight:600;color:#555;letter-spacing:.08em;text-transform:uppercase}
#grid{display:grid;grid-template-columns:repeat(4,1fr);grid-template-rows:repeat(4,1fr);
  gap:8px;min-height:0}
.cell{background:#1c1c1e;border-radius:10px;overflow:hidden;display:flex;flex-direction:column;min-height:0}
.cell img{width:100%;flex:1;object-fit:cover;display:block;min-height:0}
.info{padding:5px 7px;flex-shrink:0;background:#1c1c1e}
.names{font-size:.75rem;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.names.k{color:#30d158}.names.u{color:#ff453a}.names.empty{color:#333}
.meta{font-size:.65rem;color:#555;margin-top:1px;display:flex;gap:6px}
.count{color:#888}
#live-label{font-size:.65rem;color:#555;padding:4px 7px;flex-shrink:0;text-align:center}
</style></head><body>
<header><span id="dot"></span><span>Камера</span></header>
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

// Создаём 15 пустых ячеек для детекций
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
setInterval(()=>{cam.src='frame.jpg?t='+Date.now();},150);
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


def _draw_faces(frame, detected):
    """Рисует рамки и Unicode-подписи. Один PIL-проход на кадр."""
    if not detected:
        return
    # Рамки и залитые фоны подписей — через OpenCV (быстро)
    for bbox, name, score in detected:
        x1, y1, x2, y2 = map(int, bbox)
        color = (0, 200, 0) if name != UNKNOWN_LABEL else (0, 0, 200)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, y2 - 30), (x2, y2), color, cv2.FILLED)
    # Текст с кириллицей — один PIL-проход для всех лиц
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    for bbox, name, score in detected:
        x1, _, _, y2 = map(int, bbox)
        draw.text((x1 + 6, y2 - 27), f"{name} {score:.2f}",
                  font=_UI_FONT, fill=(255, 255, 255))
    frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


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
    # core_mask: 0=auto, 1=Core0, 2=Core1, 4=Core2, 3=Core0+1, 7=все три
    def __init__(self, path, core_mask=None):
        self.net = RKNNLite()
        if self.net.load_rknn(path) != 0:
            raise RuntimeError(f"Не удалось загрузить модель: {path}")
        kwargs = {} if core_mask is None else {"core_mask": core_mask}
        if self.net.init_runtime(**kwargs) != 0:
            raise RuntimeError("Ошибка инициализации RKNN runtime")

    def _run(self, inputs):
        return self.net.inference(inputs=inputs)

    def release(self):
        self.net.release()


class FaceDetector(_RKNNModel):
    def __init__(self, path, core_mask=None):
        global _SCRFD_INPUT, _ANCHORS
        super().__init__(path, core_mask=core_mask)
        # Определяем размер входа перебором стандартных размеров
        for candidate in [640, 480, 360, 320]:
            probe = np.zeros((1, candidate, candidate, 3), dtype=np.uint8)
            outs = self._run([probe])
            if outs is not None:
                if candidate != _SCRFD_INPUT:
                    print(f"[*] SCRFD input: модель {candidate}×{candidate}, код был {_SCRFD_INPUT}×{_SCRFD_INPUT} — исправляю")
                    _SCRFD_INPUT = candidate
                    _ANCHORS = _build_anchor_centers()
                break
        else:
            raise RuntimeError("Не удалось определить размер входа SCRFD — ни один из кандидатов не подошёл")

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
    # SCRFD → Core0, ArcFace → Core1: работают параллельно на разных ядрах NPU
    detector = FaceDetector(SCRFD_MODEL,  core_mask=1)  # NPU Core0
    encoder  = FaceEncoder(ARCFACE_MODEL, core_mask=2)  # NPU Core1

    print("[*] Загрузка базы лиц...")
    known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR, detector, encoder)

    def _open_camera():
        for idx in range(CAMERA_MAX_INDEX):
            c = cv2.VideoCapture(idx)
            if c.isOpened():
                r, _ = c.read()
                if r:
                    c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    return c, idx
            c.release()
        return None, -1

    cap, used_index = _open_camera()
    if cap is None:
        print(f"[!] Не найдена доступная камера (проверены индексы 0..{CAMERA_MAX_INDEX - 1})")
        detector.release()
        encoder.release()
        return
    print(f"[*] Камера открыта: индекс {used_index}")

    # ── Асинхронный пайплайн кодирования ────────────────────────────────────
    # Пока детектор обрабатывает кадр N на Core0,
    # энкодер параллельно кодирует лица из кадра N-1 на Core1.
    _enc_in:  queue.Queue = queue.Queue(maxsize=2)
    _enc_out: queue.Queue = queue.Queue(maxsize=2)

    def _encode_loop():
        # История сырых скоров для каждого кандидата — сглаживание по времени
        score_hist: dict = {}

        while True:
            item = _enc_in.get()
            if item is None:
                break
            frm, raw_faces = item
            results = []

            for bbox, kps in raw_faces:
                frontal, offset = _is_frontal(kps)
                if not frontal:
                    if DEBUG:
                        ts = time.strftime("%H:%M:%S")
                        print(f"[D] {ts} пропуск: боковой профиль offset={offset:.3f} (thresh={_FRONTAL_THRESH})")
                    continue

                enc, aligned = encoder.encode(frm, kps)
                if enc is None:
                    results.append((bbox, UNKNOWN_LABEL, 0.0))
                    continue

                _, _, top = identify_face(enc, known_encodings, known_names)

                if top:
                    best_cand, best_raw = top[0]
                    hist = score_hist.setdefault(best_cand, deque(maxlen=SCORE_WINDOW))
                    hist.append(best_raw)
                    avg = float(np.mean(hist))
                    name  = best_cand if avg >= RECOGNITION_THRESHOLD else UNKNOWN_LABEL
                    score = avg
                    # Не считаем незнакомцем при очень низком сходстве — это скорее не лицо
                    if name == UNKNOWN_LABEL and score < STRANGER_MIN_SCORE:
                        continue
                else:
                    name, score = UNKNOWN_LABEL, 0.0
                    if score < STRANGER_MIN_SCORE:
                        continue

                if DEBUG:
                    ts = time.strftime("%H:%M:%S")
                    cands = "  ".join(f"{n}={s:.3f}" for n, s in top)
                    status = "✓" if name != UNKNOWN_LABEL else "?"
                    n_hist = len(score_hist.get(top[0][0], [])) if top else 0
                    print(f"[D] {ts} {status} best={name} score(avg{n_hist})={score:.3f}  "
                          f"offset={offset:.3f}  [{cands}]  thresh={RECOGNITION_THRESHOLD} stranger_min={STRANGER_MIN_SCORE}")
                    if aligned is not None and WEB_PORT:
                        try:
                            debug_path = os.path.join(WEB_DIR, "aligned_last.jpg")
                            ok, buf = cv2.imencode(".jpg", aligned)
                            if ok:
                                _write_file(debug_path, buf.tobytes())
                        except OSError:
                            pass

                results.append((bbox, name, score))

            _enc_out.put(results)

    enc_thread = threading.Thread(target=_encode_loop, daemon=True)
    enc_thread.start()
    print("[*] Запуск. Нажми Q для выхода.")

    last_greeted: dict = {}
    confirm_streak: dict = {}
    stranger_streak: int = 0  # подряд кадров с незнакомцем (как CONFIRM_FRAMES для известных)
    last_stranger_streak_time: float = 0.0  # время последнего инкремента stranger_streak
    last_heartbeat = [0.0]
    last_debug_state: tuple = (frozenset(), 0)  # (known_names, unknown_count)
    last_event_time: dict = {}
    frame_count = 0
    detected: list = []
    no_frame_count = 0
    # Отложенная запись незнакомца: ждём STRANGER_CONFIRM_DELAY секунд перед фиксацией
    # (saved_frame, saved_detected, saved_time, web_names, event_key) или None
    _pending_stranger = None

    while True:
        ret, frame = cap.read()
        if not ret:
            no_frame_count += 1
            print("[!] Нет кадра с камеры")
            if no_frame_count >= NO_FRAME_RECONNECT_AFTER:
                print(f"[*] Ждём {NO_FRAME_RECONNECT_DELAY} с, затем ищем другую камеру...")
                cap.release()
                time.sleep(NO_FRAME_RECONNECT_DELAY)
                cap, used_index = _open_camera()
                if cap is None:
                    print("[!] Не удалось найти камеру после переподключения.")
                    break
                print(f"[*] Камера открыта: индекс {used_index}")
                no_frame_count = 0
            time.sleep(0.1)
            continue

        no_frame_count = 0
        frame_count += 1

        if frame_count % PROCESS_EVERY_N == 0:
            faces = detector.detect(frame)
            # Отправляем детекции в очередь кодировщика (Core1 обработает параллельно)
            try:
                _enc_in.put_nowait((frame.copy(), faces))
            except queue.Full:
                pass  # кодировщик ещё занят — пропускаем кадр

        # Забираем готовые результаты от кодировщика (не блокируем основной поток)
        try:
            face_results: list = _enc_out.get_nowait()
        except queue.Empty:
            face_results = None

        if face_results is not None:
            detected = face_results
            current_names: set = set()
            raw_unknown_count = 0

            for bbox, name, score in face_results:
                if name == UNKNOWN_LABEL:
                    raw_unknown_count += 1
                    continue
                current_names.add(name)
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

            # Стрик для незнакомца: инкремент не чаще чем STREAK_MIN_INTERVAL (3 стрика ≈ 2 с)
            now_ts = time.time()
            if raw_unknown_count > 0:
                if stranger_streak == 0 or (now_ts - last_stranger_streak_time) >= STREAK_MIN_INTERVAL:
                    stranger_streak += 1
                    last_stranger_streak_time = now_ts
                if DEBUG:
                    print(f"[D]   stranger_streak={stranger_streak}/{CONFIRM_FRAMES}")
            else:
                stranger_streak = 0
            unknown_count = raw_unknown_count if stranger_streak >= CONFIRM_FRAMES else 0

            # Сбрасываем стрик для тех, кого нет в текущем кадре
            for gone in set(confirm_streak) - current_names - {UNKNOWN_LABEL}:
                confirm_streak[gone] = 0

            current_state = (frozenset(current_names), unknown_count)
            if current_state != last_debug_state:
                web_names = list(current_names) + [UNKNOWN_LABEL] * unknown_count
                if web_names:
                    key = current_state
                    now = time.time()
                    if current_names:
                        # Опознанный человек — записываем сразу, отменяем незнакомца
                        _pending_stranger = None
                        if now - last_event_time.get(key, 0) >= WEB_EVENT_COOLDOWN:
                            last_event_time[key] = now
                            snap = frame.copy()
                            _draw_faces(snap, detected)
                            _web_add_event(web_names, snap)
                    else:
                        # Только незнакомец — ждём STRANGER_CONFIRM_DELAY секунд
                        if _pending_stranger is None:
                            _pending_stranger = (frame.copy(), list(detected), now, web_names, key)
                            if DEBUG:
                                ts = time.strftime("%H:%M:%S")
                                print(f"[D] {ts} незнакомец в ожидании ({STRANGER_CONFIRM_DELAY:.0f}с)...")
                else:
                    # Лица пропали — отменяем ожидание незнакомца
                    _pending_stranger = None
                    if DEBUG:
                        ts = time.strftime("%H:%M:%S")
                        print(f"[D] {ts} кадр {frame_count}: лиц не обнаружено")
                last_debug_state = current_state
                last_heartbeat[0] = time.time()
            elif DEBUG and not current_names and not unknown_count and time.time() - last_heartbeat[0] >= DEBUG_INTERVAL:
                ts = time.strftime("%H:%M:%S")
                print(f"[D] {ts} кадр {frame_count}: лиц не обнаружено")
                last_heartbeat[0] = time.time()

        # Проверяем таймер незнакомца (вне блока face_results — работает каждый кадр)
        if _pending_stranger is not None:
            saved_frame, saved_det, saved_time, saved_names, saved_key = _pending_stranger
            if time.time() - saved_time >= STRANGER_CONFIRM_DELAY:
                _pending_stranger = None
                now = time.time()
                if now - last_event_time.get(saved_key, 0) >= WEB_EVENT_COOLDOWN:
                    last_event_time[saved_key] = now
                    snap = saved_frame
                    _draw_faces(snap, saved_det)
                    _web_add_event(saved_names, snap)
                if DEBUG:
                    ts = time.strftime("%H:%M:%S")
                    print(f"[D] {ts} незнакомец подтверждён после {STRANGER_CONFIRM_DELAY:.0f}с → событие записано")

        _draw_faces(frame, detected)

        if WEB_PORT:
            with _latest_frame_lock:
                _latest_frame_bgr = frame

        if SHOW_DISPLAY:
            cv2.imshow("Faces", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    _enc_in.put(None)  # сигнал завершения энкодеру
    enc_thread.join(timeout=2)
    if cap is not None:
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
