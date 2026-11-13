"""Reference-based quality metrics for eval fixtures that ship with a clean plate.

Fixtures are synthetic, so the clean (unwatermarked) video is known. That turns the eval
from "did the job succeed" into numbers a person does not have to eyeball:

- `psnr_region`   fidelity of the reconstructed watermark area versus the clean plate
- `psnr_outside`  the rest of the frame must come back untouched (ROI/composite guard)
- `ssim_region`   structural similarity inside the watermark area
- `flicker_region` temporal difference error inside the watermark area versus the clean
                  plate (frame-to-frame changes the remover invented)
- `residual_detected` whether the static detector still finds an overlay in the output

The judgment thresholds are deliberately conservative and only apply to AI providers;
`local_fallback` copies are never auto-passed.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import cv2
import numpy as np

REGION_MEDIAN_DIFF = 32  # static/translucent overlays: consistent per-pixel difference
REGION_STRONG_DIFF = 64  # moving overlays: a large difference in any single frame
REGION_DILATE_PX = 3
PSNR_CAP = 99.0

PASS_PSNR_REGION = 28.0
DEGRADED_PSNR_REGION = 22.0
PASS_PSNR_OUTSIDE = 35.0  # absolute floor for untouched pixels ...
OUTSIDE_TOLERANCE_DB = 3.0  # ... relaxed to (input's own outside PSNR - 3 dB) for lossy inputs
PASS_SSIM_REGION = 0.80
PASS_FLICKER_REGION = 6.0


def read_frames(path: Path, *, limit: int = 4096) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return []
    frames: list[np.ndarray] = []
    try:
        while len(frames) < limit:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frames.append(frame)
    finally:
        capture.release()
    return frames


def load_region(region_path: Path) -> np.ndarray | None:
    image = cv2.imread(str(region_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return image > 127


def watermark_region(input_path: Path, clean_path: Path) -> np.ndarray | None:
    """Pixels where the watermarked input differs from the clean plate.

    Used when no ground-truth region image ships with the case. Codec noise on the
    (lossy) input is tolerated by voting: a pixel counts when its median difference
    across frames is clear, or when any single frame differs strongly (a moving overlay
    only touches a pixel in a few frames).
    """
    dirty = read_frames(input_path)
    clean = read_frames(clean_path)
    count = min(len(dirty), len(clean))
    if count == 0:
        return None
    diffs = []
    for index in range(count):
        a = dirty[index].astype(np.int16)
        b = _match_size(clean[index], dirty[index]).astype(np.int16)
        diffs.append(np.abs(a - b).max(axis=2))
    stacked = np.stack(diffs, axis=0)
    region = (np.median(stacked, axis=0) > REGION_MEDIAN_DIFF) | (stacked.max(axis=0) > REGION_STRONG_DIFF)
    if REGION_DILATE_PX > 0:
        kernel = np.ones((2 * REGION_DILATE_PX + 1, 2 * REGION_DILATE_PX + 1), np.uint8)
        region = cv2.dilate(region.astype(np.uint8), kernel) > 0
    return region


def compare_to_clean(output_path: Path, clean_path: Path, region: np.ndarray) -> dict[str, object]:
    output = read_frames(output_path)
    clean = read_frames(clean_path)
    count = min(len(output), len(clean))
    if count == 0:
        return {"frames_compared": 0, "error": "no comparable frames"}
    if not region.any():
        return {"frames_compared": count, "error": "empty watermark region"}
    outside = ~region
    region_mse: list[float] = []
    outside_mse: list[float] = []
    ssim_values: list[float] = []
    flicker: list[float] = []
    prev_out: np.ndarray | None = None
    prev_clean: np.ndarray | None = None
    for index in range(count):
        out = _match_size(output[index], clean[index]).astype(np.float32)
        ref = clean[index].astype(np.float32)
        squared = ((out - ref) ** 2).mean(axis=2)
        region_mse.append(float(squared[region].mean()))
        if outside.any():
            outside_mse.append(float(squared[outside].mean()))
        out_gray = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        ref_gray = cv2.cvtColor(ref.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        ssim_values.append(_ssim_region(out_gray, ref_gray, region))
        if prev_out is not None and prev_clean is not None:
            out_delta = np.abs(out_gray - prev_out)
            ref_delta = np.abs(ref_gray - prev_clean)
            flicker.append(float(np.abs(out_delta - ref_delta)[region].mean()))
        prev_out, prev_clean = out_gray, ref_gray
    metrics: dict[str, object] = {
        "frames_compared": count,
        "region_ratio": round(float(region.mean()), 6),
        "psnr_region": round(_psnr(float(np.mean(region_mse))), 3),
        "psnr_outside": round(_psnr(float(np.mean(outside_mse))), 3) if outside_mse else None,
        "ssim_region": round(float(np.mean(ssim_values)), 4),
        "flicker_region": round(float(np.mean(flicker)), 3) if flicker else 0.0,
    }
    return metrics


def residual_overlay(output_path: Path) -> dict[str, object]:
    """Re-run the static detector on the output: a surviving overlay is a failed removal."""
    from wm_platform.watermark_detect import detect_static_overlay

    with tempfile.TemporaryDirectory(prefix="cleanplate-residual-") as tmp:
        result = detect_static_overlay(output_path, Path(tmp) / "residual.mask.mp4")
    detected = result.mask_video_path is not None and result.confidence_level in {"high", "medium"}
    return {
        "residual_detected": bool(detected),
        "residual_mask_ratio": float(result.mask_ratio),
        "residual_scene_type": result.scene_type,
    }


def evaluate_output(
    output_path: Path,
    input_path: Path,
    clean_path: Path,
    *,
    region_path: Path | None = None,
) -> dict[str, object]:
    region = load_region(region_path) if region_path is not None else None
    region_source = "ground_truth"
    if region is None:
        region = watermark_region(input_path, clean_path)
        region_source = "diff"
    if region is None:
        return {"frames_compared": 0, "error": "could not derive watermark region"}
    metrics = compare_to_clean(output_path, clean_path, region)
    metrics["region_source"] = region_source
    baseline = compare_to_clean(input_path, clean_path, region)
    metrics["psnr_outside_input"] = baseline.get("psnr_outside")
    metrics["psnr_region_input"] = baseline.get("psnr_region")
    metrics.update(residual_overlay(output_path))
    return metrics


def auto_judgment(metrics: dict[str, object]) -> str:
    if metrics.get("error") or not metrics.get("frames_compared"):
        return "pending"
    psnr_region = _as_float(metrics.get("psnr_region"))
    psnr_outside = _as_float(metrics.get("psnr_outside"))
    psnr_outside_input = _as_float(metrics.get("psnr_outside_input"))
    ssim_region = _as_float(metrics.get("ssim_region"))
    flicker_region = _as_float(metrics.get("flicker_region"))
    residual = bool(metrics.get("residual_detected"))
    if psnr_region is None:
        return "pending"
    if residual or psnr_region < DEGRADED_PSNR_REGION:
        return "fail"
    outside_floor = PASS_PSNR_OUTSIDE
    if psnr_outside_input is not None:
        outside_floor = min(outside_floor, psnr_outside_input - OUTSIDE_TOLERANCE_DB)
    outside_ok = psnr_outside is None or psnr_outside >= outside_floor
    ssim_ok = ssim_region is None or ssim_region >= PASS_SSIM_REGION
    flicker_ok = flicker_region is None or flicker_region <= PASS_FLICKER_REGION
    if psnr_region >= PASS_PSNR_REGION and outside_ok and ssim_ok and flicker_ok:
        return "pass"
    return "degraded"


def _as_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _psnr(mse: float) -> float:
    if mse <= 1e-9:
        return PSNR_CAP
    return min(PSNR_CAP, 10.0 * math.log10((255.0**2) / mse))


def _match_size(frame: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if frame.shape[:2] == reference.shape[:2]:
        return frame
    return cv2.resize(frame, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_AREA)


def _ssim_region(a: np.ndarray, b: np.ndarray, region: np.ndarray) -> float:
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    blur = lambda img: cv2.GaussianBlur(img, (7, 7), 1.5)  # noqa: E731
    mu_a = blur(a)
    mu_b = blur(b)
    sigma_a = blur(a * a) - mu_a * mu_a
    sigma_b = blur(b * b) - mu_b * mu_b
    sigma_ab = blur(a * b) - mu_a * mu_b
    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a + sigma_b + c2)
    ssim_map = numerator / np.maximum(denominator, 1e-9)
    return float(ssim_map[region].mean())
