"""Face database with hot-reload support."""

import threading
from pathlib import Path

import cv2
import numpy as np

from config import IMAGE_SUFFIXES


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


# ── Загрузка из каталога ─────────────────────────────────────────────────────
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


# ── FaceDB ───────────────────────────────────────────────────────────────────
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
