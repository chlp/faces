"""
Downloads InsightFace ONNX models and converts them to RKNN for RK3588.

Models from antelopev2 package:
  scrfd_10g_bnkps.onnx -> /output/scrfd.rknn   (SCRFD-10G, face detection)
  glintr100.onnx       -> /output/arcface.rknn (ArcFace ResNet100 / Glint360K, 512-dim embedding)

ArcFace ResNet100 (~24 GFLOPs) is significantly more accurate than ResNet50 (~4 GFLOPs),
trained on Glint360K (360M pairs, 17M unique faces).
"""

import urllib.request
import zipfile
import os
from pathlib import Path

import onnx
from rknn.api import RKNN


def fix_dynamic_shape(onnx_path: str, shape: list) -> str:
    """Replace dynamic axes with static ones in an ONNX model."""
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

# ── Download models ───────────────────────────────────────────────────────────
PACK_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip"
PACK_ZIP  = Path("/tmp/antelopev2.zip")
PACK_DIR  = Path("/tmp/antelopev2")

print("[*] Downloading insightface antelopev2 (SCRFD-10G + ArcFace ResNet100/Glint360K)...")
urllib.request.urlretrieve(PACK_URL, PACK_ZIP)

with zipfile.ZipFile(PACK_ZIP) as z:
    z.extractall(PACK_DIR)

# ONNX paths inside the archive (names may vary — search dynamically)
def find_onnx(directory, keyword):
    for p in Path(directory).rglob("*.onnx"):
        if keyword.lower() in p.name.lower():
            return str(p)
    return None

det_onnx  = find_onnx(PACK_DIR, "scrfd_10g") or find_onnx(PACK_DIR, "det_10g") or find_onnx(PACK_DIR, "det_2.5g")
face_onnx = find_onnx(PACK_DIR, "glintr100") or find_onnx(PACK_DIR, "w600k_r50") or find_onnx(PACK_DIR, "w600k")

if not det_onnx:
    raise FileNotFoundError("det_*.onnx not found in archive. Files: " +
                            str(list(PACK_DIR.rglob("*.onnx"))))
if not face_onnx:
    raise FileNotFoundError("*mbf*.onnx not found in archive. Files: " +
                            str(list(PACK_DIR.rglob("*.onnx"))))

print(f"[+] Detection:    {det_onnx}")
print(f"[+] Recognition: {face_onnx}")


# ── Conversion ────────────────────────────────────────────────────────────────
def convert(onnx_path: str, output_path: str, label: str, input_size: int):
    print(f"\n[*] Converting {label}...")
    rknn = RKNN(verbose=False)

    # target_platform='rk3588', normalization baked into model (input — uint8 RGB [0-255])
    rknn.config(
        target_platform="rk3588",
        optimization_level=3,
        mean_values=[[127.5, 127.5, 127.5]],
        std_values=[[128.0, 128.0, 128.0]],
    )

    # Fix dynamic shape before conversion
    fixed = fix_dynamic_shape(onnx_path, [1, 3, input_size, input_size])
    ret = rknn.load_onnx(model=fixed)
    assert ret == 0, f"ONNX load error: {onnx_path}"

    ret = rknn.build(do_quantization=False)
    assert ret == 0, "Build error"

    ret = rknn.export_rknn(output_path)
    assert ret == 0, f"Export error: {output_path}"

    rknn.release()
    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"[+] Saved: {output_path}  ({size_mb:.1f} MB)")


convert(det_onnx,  str(OUTPUT_DIR / "scrfd.rknn"),   "SCRFD-10G → scrfd.rknn",          640)
convert(face_onnx, str(OUTPUT_DIR / "arcface.rknn"), "ArcFace ResNet100 → arcface.rknn", 112)

print("\n[✓] Done! Copy models/scrfd.rknn and models/arcface.rknn to Orange Pi.")
