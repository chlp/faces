#!/usr/bin/env python3
"""
Face recognition on NPU (RKNN) — Orange Pi 5 Max / RK3588.
Detection: SCRFD-10G  |  Encoding: ArcFace ResNet100 (Glint360K)
Download models: bash download_models.sh
"""

import os
import queue
import signal
import threading
import time
from collections import deque

import cv2
import numpy as np

from config import (
    GREETINGS, RELOAD_CHECK_S, SCRFD_MODEL, ARCFACE_MODEL, UNKNOWN_LABEL,
    parse_args,
)
from detection import FaceDetector, FaceEncoder, _is_frontal
from facedb import FaceDB, identify_face
from store import EventStore
from tracker import FaceTracker
from ui import draw_faces, open_camera, speak
from web import WebServer


# ── Event dispatcher ─────────────────────────────────────────────────────────
def _dispatch(ev, cfg, event_store, web):
    if ev[0] == "greet" and not cfg.no_tts:
        greeting = GREETINGS[cfg.lang].format(ev[1])
        print(f"[>] {greeting}")
        speak(greeting, cfg.lang)
    elif ev[0] == "web_event":
        _, names, snap, det = ev
        draw_faces(snap, det)
        ok, buf = cv2.imencode(".jpg", snap, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            event_store.add(names, buf.tobytes())
        if web:
            web.notify_detection()


# ── Main loop ───────────────────────────────────────────────────────────────
def main():
    cfg = parse_args()
    os.makedirs(cfg.data_dir, exist_ok=True)

    event_store = EventStore(cfg.db_path)
    web = WebServer(cfg.web_port, event_store) if cfg.web_port else None

    print("[*] Initializing NPU...")
    detector = FaceDetector(SCRFD_MODEL, core_mask=1)
    encoder = FaceEncoder(ARCFACE_MODEL, core_mask=2)
    face_db = FaceDB(cfg.known_faces_dir, detector, encoder)
    tracker = FaceTracker(cfg)

    # restore last_greeted from DB (avoid re-greeting after restart)
    tracker.last_greeted.update(
        event_store.last_greeted_times(cfg.greet_cooldown)
    )

    cap = None
    while cap is None:
        cap, _ = open_camera(cfg)
        if cap is None:
            print("[!] Camera not found, retrying in 5 s...")
            time.sleep(5)

    # ── async encode pipeline (Core1) ────────────────────────────────────
    _enc_in: queue.Queue = queue.Queue(maxsize=2)
    _enc_out: queue.Queue = queue.Queue()

    def _encode_loop():
        score_hist: dict = {}
        while True:
            item = _enc_in.get()
            if item is None:
                break
            if item == "pause":
                _enc_out.put("paused")
                _enc_in.get()  # wait for "resume"
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
                            f"profile offset={offset:.3f}"
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
        print(f"\n[*] Signal {sig}, shutting down...")

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    print("[*] Starting. Q to quit.")
    frame_count = 0
    no_frame_count = 0
    last_reload_check = time.time()

    # ── main loop ────────────────────────────────────────────────────────
    while _running:
        ret, frame = cap.read()
        if not ret:
            no_frame_count += 1
            if no_frame_count >= cfg.no_frame_reconnect_after:
                print("[*] Reconnecting camera...")
                cap.release()
                time.sleep(cfg.no_frame_reconnect_delay)
                cap, _ = open_camera(cfg)
                while cap is None:
                    print("[!] Camera not found, retrying in 5 s...")
                    time.sleep(5)
                    cap, _ = open_camera(cfg)
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

        draw_faces(frame, tracker.detected)

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
    print("[*] Done.")


if __name__ == "__main__":
    main()
