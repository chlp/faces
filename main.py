#!/usr/bin/env python3
"""
Распознавание лиц на NPU (RKNN) — Orange Pi 5 Max / RK3588.
Детекция: SCRFD-2.5G  |  Энкодинг: MobileFaceNet/ArcFace
Скачать модели: bash download_models.sh
"""

import time
import subprocess
import threading
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
PROCESS_EVERY_N       = 2     # обрабатывать каждый N-й кадр
CONFIRM_FRAMES        = 3     # сколько кадров подряд нужно видеть человека перед приветствием
SHOW_DISPLAY          = False # показывать окно (требует X-сервер/дисплей)
DEBUG                 = True  # подробный вывод в консоль
DEBUG_INTERVAL        = 5.0   # секунд между пульсом (если нет детекций)

# Порог распознавания — косинусное сходство (0..1, выше = строже)
RECOGNITION_THRESHOLD = 0.55

GREETINGS     = {"ru": "Привет, {}!", "en": "Hello, {}!"}
LANG          = "ru"
UNKNOWN_LABEL = "Незнакомец"
ESPEAK_ARGS   = {"ru": ["-v", "ru", "-s", "140"], "en": ["-v", "en"]}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# ── SCRFD внутренние параметры ───────────────────────────────────────────────
_SCRFD_INPUT  = 640
_STRIDES      = [8, 16, 32]
_NUM_ANCHORS  = 2
_SCORE_THRESH = 0.50
_NMS_THRESH   = 0.40

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
        fs = _SCRFD_INPUT // stride
        gy, gx = np.mgrid[0:fs, 0:fs]
        c = np.stack([gx * stride, gy * stride], axis=-1).reshape(-1, 2)
        # Каждую точку сетки повторяем _NUM_ANCHORS раз
        centers[stride] = np.repeat(c, _NUM_ANCHORS, axis=0).astype(np.float32)
    return centers

_ANCHORS = _build_anchor_centers()


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
        scores = outputs[i + 0].flatten()
        bboxes = outputs[i + n].reshape(-1, 4)
        kps    = outputs[i + n * 2].reshape(-1, 10)
        ac     = _ANCHORS[stride]

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
        inp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        inp = (inp - 127.5) / 128.0
        inp = inp.transpose(2, 0, 1)[np.newaxis]        # 1×3×640×640
        return _decode_scrfd(self._run([inp]), scale, pad)


class FaceEncoder(_RKNNModel):
    def encode(self, frame, kps):
        """Выравнивает лицо по 5 точкам, возвращает L2-норм. эмбеддинг (512,) или None."""
        M, _ = cv2.estimateAffinePartial2D(kps, _ARCFACE_TPL, method=cv2.LMEDS)
        if M is None:
            return None
        aligned = cv2.warpAffine(frame, M, (112, 112), borderValue=0.0)
        inp = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
        inp = (inp - 127.5) / 128.0
        inp = inp.transpose(2, 0, 1)[np.newaxis]        # 1×3×112×112
        emb = self._run([inp])[0].flatten()
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else None


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
            enc = encoder.encode(img, kps)
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
        return UNKNOWN_LABEL, 0.0
    # Косинусное сходство (оба вектора уже L2-нормализованы → просто dot product)
    sims = known_encodings @ encoding
    best = int(np.argmax(sims))
    score = float(sims[best])
    name = known_names[best] if score >= RECOGNITION_THRESHOLD else UNKNOWN_LABEL
    return name, score


# ── Главный цикл ─────────────────────────────────────────────────────────────
def main():
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
            # Сбрасываем счётчик для тех, кого нет в текущем кадре
            for gone in set(confirm_streak) - {UNKNOWN_LABEL}:
                confirm_streak[gone] = 0
            for bbox, kps in faces:
                enc = encoder.encode(frame, kps)
                if enc is not None:
                    name, score = identify_face(enc, known_encodings, known_names)
                else:
                    name, score = UNKNOWN_LABEL, 0.0
                detected.append((bbox, name))
                current_names.add(name)

                if name != UNKNOWN_LABEL:
                    confirm_streak[name] = confirm_streak.get(name, 0) + 1
                    if confirm_streak[name] >= CONFIRM_FRAMES:
                        now = time.time()
                        if now - last_greeted.get(name, 0) > GREET_COOLDOWN:
                            last_greeted[name] = now
                            greeting = GREETINGS[LANG].format(name)
                            print(f"[>] {greeting}")
                            speak(greeting)

            if DEBUG:
                now = time.time()
                if current_names != last_debug_names:
                    if current_names:
                        print(f"[D] Кадр {frame_count}: {sorted(current_names)}")
                    else:
                        print(f"[D] Кадр {frame_count}: лиц не обнаружено")
                    last_debug_names = current_names
                    last_heartbeat[0] = now
                elif not current_names and now - last_heartbeat[0] >= DEBUG_INTERVAL:
                    print(f"[D] Кадр {frame_count}: лиц не обнаружено")
                    last_heartbeat[0] = now

        for bbox, name in detected:
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 200, 0) if name != UNKNOWN_LABEL else (0, 0, 200)
            draw_box(frame, x1, y1, x2, y2, name, color)

        if SHOW_DISPLAY:
            cv2.imshow("Faces", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if SHOW_DISPLAY:
        cv2.destroyAllWindows()
    detector.release()
    encoder.release()
    print("[*] Завершено.")


if __name__ == "__main__":
    main()
