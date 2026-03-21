"""SCRFD detection + ArcFace encoding (RKNN wrappers)."""

import math
import time

import cv2
import numpy as np
from rknnlite.api import RKNNLite

import config as cfg


# ── Utilities ────────────────────────────────────────────────────────────────
def letterbox(img, size=None, fill=(114, 114, 114)):
    """Scale with aspect ratio preserved and pad to square."""
    if size is None:
        size = cfg._SCRFD_INPUT
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh))
    pad_h, pad_w = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), fill, dtype=np.uint8)
    out[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized
    return out, scale, (pad_w, pad_h)


# ── Anchors ──────────────────────────────────────────────────────────────────
def _build_anchor_centers(input_size=None):
    if input_size is None:
        input_size = cfg._SCRFD_INPUT
    centers = {}
    for stride in cfg._STRIDES:
        fs = math.ceil(input_size / stride)
        gy, gx = np.mgrid[0:fs, 0:fs]
        c = np.stack([gx * stride, gy * stride], axis=-1).reshape(-1, 2)
        centers[stride] = np.repeat(c, cfg._NUM_ANCHORS, axis=0).astype(np.float32)
    return centers


_ANCHORS = _build_anchor_centers()


# ── SCRFD post-processing ────────────────────────────────────────────────────
def _is_frontal(kps) -> tuple:
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_dist = abs(right_eye[0] - left_eye[0])
    if eye_dist < 1:
        return False, 1.0
    eye_center_x = (left_eye[0] + right_eye[0]) / 2
    offset = abs(nose[0] - eye_center_x) / eye_dist
    return offset < cfg._FRONTAL_THRESH, offset


def _decode_scrfd(outputs, scale, pad):
    pad_w, pad_h = pad
    all_boxes, all_scores, all_kps = [], [], []
    n = len(cfg._STRIDES)
    for i, stride in enumerate(cfg._STRIDES):
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
        mask = scores > cfg._SCORE_THRESH
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
    idxs = cv2.dnn.NMSBoxes(xywh, scores.tolist(), cfg._SCORE_THRESH, cfg._NMS_THRESH)
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


# ── RKNN wrappers ────────────────────────────────────────────────────────────
class _RKNNModel:
    def __init__(self, path, core_mask=None, metrics=None):
        self._metrics = metrics
        self.net = RKNNLite()
        if self.net.load_rknn(path) != 0:
            raise RuntimeError(f"Failed to load model: {path}")
        kwargs = {} if core_mask is None else {"core_mask": core_mask}
        if self.net.init_runtime(**kwargs) != 0:
            raise RuntimeError("RKNN init_runtime error")

    def _run(self, inputs):
        return self.net.inference(inputs=inputs)

    def release(self):
        self.net.release()


class FaceDetector(_RKNNModel):
    def __init__(self, path, core_mask=None, metrics=None):
        global _ANCHORS
        super().__init__(path, core_mask=core_mask, metrics=metrics)
        for candidate in [640, 480, 360, 320]:
            probe = np.zeros((1, candidate, candidate, 3), dtype=np.uint8)
            outs = self._run([probe])
            if outs is not None:
                if candidate != cfg._SCRFD_INPUT:
                    print(f"[*] SCRFD: {candidate}x{candidate}")
                    cfg._SCRFD_INPUT = candidate
                    _ANCHORS = _build_anchor_centers(candidate)
                break
        else:
            raise RuntimeError("Failed to determine SCRFD input size")

    def detect(self, frame):
        t0 = time.perf_counter()
        img, scale, pad = letterbox(frame)
        inp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t_npu0 = time.perf_counter()
        outs = self._run([inp[np.newaxis]])
        t_npu1 = time.perf_counter()
        result = _decode_scrfd(outs, scale, pad)
        t1 = time.perf_counter()
        if self._metrics:
            self._metrics.record_detection(
                t_npu1 - t_npu0, t1 - t_npu1, t1 - t0
            )
        return result


class FaceEncoder(_RKNNModel):
    def __init__(self, path, core_mask=None, metrics=None):
        super().__init__(path, core_mask=core_mask, metrics=metrics)

    def encode(self, frame, kps):
        t0 = time.perf_counter()
        M, _ = cv2.estimateAffinePartial2D(
            kps, cfg._ARCFACE_TPL, method=cv2.LMEDS
        )
        if M is None:
            return None, None
        aligned = cv2.warpAffine(frame, M, (112, 112), borderValue=0.0)
        inp = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        t_npu0 = time.perf_counter()
        emb = self._run([inp[np.newaxis]])[0].flatten()
        t_npu1 = time.perf_counter()
        if self._metrics:
            self._metrics.record_encode_face(t_npu1 - t_npu0, t_npu0 - t0)
        norm = np.linalg.norm(emb)
        enc = emb / norm if norm > 0 else None
        return enc, aligned
