"""
Скачивает ONNX-модели InsightFace и конвертирует их в RKNN для RK3588.

Модели из пакета buffalo_s:
  det_500m.onnx  → /output/scrfd.rknn   (SCRFD-500M, детекция лиц)
  w600k_mbf.onnx → /output/arcface.rknn (MobileFaceNet, 512-мерный эмбеддинг)
"""

import urllib.request
import zipfile
import os
from pathlib import Path

import onnx
from rknn.api import RKNN


def fix_dynamic_shape(onnx_path: str, shape: list) -> str:
    """Заменяет динамические оси на статические в ONNX-модели."""
    model = onnx.load(onnx_path)
    for inp in model.graph.input:
        dims = inp.type.tensor_type.shape.dim
        for dim, val in zip(dims, shape):
            if dim.dim_param or dim.dim_value <= 0:
                dim.ClearField("dim_param")
                dim.dim_value = val
    fixed_path = onnx_path + ".fixed.onnx"
    onnx.save(model, fixed_path)
    return fixed_path

OUTPUT_DIR = Path("/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Загрузка моделей ──────────────────────────────────────────────────────────
PACK_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"
PACK_ZIP  = Path("/tmp/buffalo_s.zip")
PACK_DIR  = Path("/tmp/buffalo_s")

print("[*] Скачивание insightface buffalo_s (SCRFD-500M + MobileFaceNet)...")
urllib.request.urlretrieve(PACK_URL, PACK_ZIP)

with zipfile.ZipFile(PACK_ZIP) as z:
    z.extractall(PACK_DIR)

# Пути к ONNX внутри архива (имена могут отличаться — ищем динамически)
def find_onnx(directory, keyword):
    for p in Path(directory).rglob("*.onnx"):
        if keyword.lower() in p.name.lower():
            return str(p)
    return None

det_onnx  = find_onnx(PACK_DIR, "det_500m") or find_onnx(PACK_DIR, "det")
face_onnx = find_onnx(PACK_DIR, "mbf") or find_onnx(PACK_DIR, "w600k")

if not det_onnx:
    raise FileNotFoundError("Не нашёл det_*.onnx в архиве. Файлы: " +
                            str(list(PACK_DIR.rglob("*.onnx"))))
if not face_onnx:
    raise FileNotFoundError("Не нашёл *mbf*.onnx в архиве. Файлы: " +
                            str(list(PACK_DIR.rglob("*.onnx"))))

print(f"[+] Детекция:     {det_onnx}")
print(f"[+] Распознавание: {face_onnx}")


# ── Конвертация ───────────────────────────────────────────────────────────────
def convert(onnx_path: str, output_path: str, label: str, input_size: int):
    print(f"\n[*] Конвертация {label}...")
    rknn = RKNN(verbose=False)

    # target_platform='rk3588', без квантизации (не нужна калибровочная выборка)
    rknn.config(target_platform="rk3588", optimization_level=3)

    # Фиксируем динамический shape перед конвертацией
    fixed = fix_dynamic_shape(onnx_path, [1, 3, input_size, input_size])
    ret = rknn.load_onnx(model=fixed)
    assert ret == 0, f"Ошибка загрузки ONNX: {onnx_path}"

    ret = rknn.build(do_quantization=False)
    assert ret == 0, "Ошибка build"

    ret = rknn.export_rknn(output_path)
    assert ret == 0, f"Ошибка экспорта: {output_path}"

    rknn.release()
    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"[+] Сохранено: {output_path}  ({size_mb:.1f} МБ)")


convert(det_onnx,  str(OUTPUT_DIR / "scrfd.rknn"),   "SCRFD-500M → scrfd.rknn",   640)
convert(face_onnx, str(OUTPUT_DIR / "arcface.rknn"), "MobileFaceNet → arcface.rknn", 112)

print("\n[✓] Готово! Скопируй models/scrfd.rknn и models/arcface.rknn на Orange Pi.")
