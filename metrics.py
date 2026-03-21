"""Runtime metrics for /health (latency, FPS, RAM, CPU, cold start)."""

from __future__ import annotations

import os
import threading
import time
from collections import deque

# Optional: better CPU% / cross-platform RAM
try:
    import psutil

    _PSUTIL = True
except ImportError:
    _PSUTIL = False


def _rss_bytes_linux() -> int | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return int(parts[1]) * 1024  # kB -> bytes
    except OSError:
        pass
    return None


class Metrics:
    """Thread-safe rolling stats; background sampler for CPU%."""

    def __init__(self, rolling: int = 120):
        self._lock = threading.Lock()
        self._rolling = rolling
        self._t0 = time.perf_counter()

        self._det_total = deque(maxlen=rolling)
        self._det_npu = deque(maxlen=rolling)
        self._det_post = deque(maxlen=rolling)
        self._enc_npu = deque(maxlen=rolling)
        self._enc_prep = deque(maxlen=rolling)
        self._e2e = deque(maxlen=rolling)

        self.encode_queue_drops = 0
        self.models_ready_ms: float | None = None
        self.first_detect_ms: float | None = None
        self.first_encode_batch_ms: float | None = None

        self._scrfd_bytes: int | None = None
        self._arcface_bytes: int | None = None

        self._camera_frames = 0
        self._detect_runs = 0
        self._fps_t0 = time.perf_counter()
        self._last_cam_fps = 0.0
        self._last_pipe_fps = 0.0
        self._encode_faces_window = 0
        self._last_encode_faces_fps = 0.0

        self._cpu_percent: float | None = None
        self._ram_bytes: int | None = None
        self._proc = None
        if _PSUTIL:
            try:
                self._proc = psutil.Process()
                self._proc.cpu_percent(interval=None)
            except Exception:
                self._proc = None

        self._sampler_stop = threading.Event()
        self._sampler = threading.Thread(target=self._sample_loop, daemon=True)
        self._sampler.start()

    def shutdown(self):
        self._sampler_stop.set()

    def _sample_loop(self):
        while not self._sampler_stop.wait(timeout=1.0):
            now = time.perf_counter()
            with self._lock:
                dt = now - self._fps_t0
                if dt > 0:
                    self._last_cam_fps = self._camera_frames / dt
                    self._last_pipe_fps = self._detect_runs / dt
                    self._last_encode_faces_fps = self._encode_faces_window / dt
                self._camera_frames = 0
                self._detect_runs = 0
                self._encode_faces_window = 0
                self._fps_t0 = now

                if _PSUTIL and self._proc is not None:
                    try:
                        self._cpu_percent = self._proc.cpu_percent(interval=None)
                        self._ram_bytes = self._proc.memory_info().rss
                    except Exception:
                        pass
                else:
                    r = _rss_bytes_linux()
                    if r is not None:
                        self._ram_bytes = r

    def set_model_paths(self, scrfd_path: str, arcface_path: str) -> None:
        try:
            self._scrfd_bytes = os.path.getsize(scrfd_path)
            self._arcface_bytes = os.path.getsize(arcface_path)
        except OSError:
            pass

    def mark_models_ready(self) -> None:
        with self._lock:
            self.models_ready_ms = (time.perf_counter() - self._t0) * 1000.0

    def record_detection(self, npu_s: float, post_s: float, total_s: float) -> None:
        el = (time.perf_counter() - self._t0) * 1000.0
        with self._lock:
            self._det_npu.append(npu_s * 1000.0)
            self._det_post.append(post_s * 1000.0)
            self._det_total.append(total_s * 1000.0)
            if self.first_detect_ms is None:
                self.first_detect_ms = el

    def record_encode_face(self, npu_s: float, prep_s: float) -> None:
        with self._lock:
            self._enc_npu.append(npu_s * 1000.0)
            self._enc_prep.append(prep_s * 1000.0)
            self._encode_faces_window += 1

    def record_e2e(self, dt_s: float) -> None:
        with self._lock:
            self._e2e.append(dt_s * 1000.0)
            if self.first_encode_batch_ms is None:
                self.first_encode_batch_ms = (
                    time.perf_counter() - self._t0
                ) * 1000.0

    def tick_camera_frame(self) -> None:
        with self._lock:
            self._camera_frames += 1

    def tick_detect_run(self) -> None:
        with self._lock:
            self._detect_runs += 1

    def drop_encode_queue(self) -> None:
        with self._lock:
            self.encode_queue_drops += 1

    def snapshot(
        self,
        encode_queue_size: int,
        encode_queue_max: int,
    ) -> dict:
        with self._lock:
            cam_fps = self._last_cam_fps
            pipe_fps = self._last_pipe_fps

            def _avg(d: deque) -> float | None:
                if not d:
                    return None
                return sum(d) / len(d)

            det_ms = _avg(self._det_total)
            emb_ms = None
            if self._enc_npu:
                emb_ms = (
                    sum(self._enc_npu) + sum(self._enc_prep)
                ) / len(self._enc_npu)

            models_total = None
            if self._scrfd_bytes is not None and self._arcface_bytes is not None:
                models_total = self._scrfd_bytes + self._arcface_bytes

            ram_mb = None
            if self._ram_bytes is not None:
                ram_mb = round(self._ram_bytes / (1024 * 1024), 1)

            cold = {
                "models_ready_ms": round(self.models_ready_ms, 1)
                if self.models_ready_ms is not None
                else None,
                "first_detect_ms": round(self.first_detect_ms, 1)
                if self.first_detect_ms is not None
                else None,
                "first_encode_batch_done_ms": round(
                    self.first_encode_batch_ms, 1
                )
                if self.first_encode_batch_ms is not None
                else None,
            }

            npu_ms_per_s_est = None
            if (
                self._det_npu
                and det_ms is not None
                and pipe_fps is not None
            ):
                dn = sum(self._det_npu) / len(self._det_npu)
                en = (
                    sum(self._enc_npu) / len(self._enc_npu)
                    if self._enc_npu
                    else 0.0
                )
                npu_ms_per_s_est = round(
                    dn * pipe_fps + en * self._last_encode_faces_fps,
                    1,
                )

            return {
                "detection_ms_avg": round(det_ms, 2) if det_ms is not None else None,
                "detection_npu_ms_avg": round(_avg(self._det_npu), 2)
                if self._det_npu
                else None,
                "detection_post_ms_avg": round(_avg(self._det_post), 2)
                if self._det_post
                else None,
                "embedding_ms_avg": round(emb_ms, 2) if emb_ms is not None else None,
                "embedding_npu_ms_avg": round(_avg(self._enc_npu), 2)
                if self._enc_npu
                else None,
                "embedding_preprocess_ms_avg": round(_avg(self._enc_prep), 2)
                if self._enc_prep
                else None,
                "end_to_end_ms_avg": round(_avg(self._e2e), 2)
                if self._e2e
                else None,
                "pipeline_fps": round(pipe_fps, 2),
                "camera_fps": round(cam_fps, 2),
                "encode_faces_per_s": round(self._last_encode_faces_fps, 2),
                "npu_inference_ms_per_s_est": npu_ms_per_s_est,
                "cpu_percent": round(self._cpu_percent, 1)
                if self._cpu_percent is not None
                else None,
                "ram_mb": ram_mb,
                "encode_queue_drops": self.encode_queue_drops,
                "encode_queue_size": encode_queue_size,
                "encode_queue_max": encode_queue_max,
                "models_bytes": {
                    "scrfd": self._scrfd_bytes,
                    "arcface": self._arcface_bytes,
                    "total": models_total,
                },
                "cold_start_ms": cold,
                "samples_window": len(self._det_total),
                "psutil": _PSUTIL,
            }
