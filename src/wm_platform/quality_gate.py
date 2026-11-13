"""Post-inpaint checks that run on every AI output before a job is marked succeeded.

Two failure modes are common enough to check on every job:

- the inpainter returned the input unchanged. ComfyUI-MiniMax-Remover catches every
  exception (CUDA OOM included) and returns the original frames, so the prompt
  "succeeds" with the watermark still there;
- the overlay is still visible (mask too tight, translucent edges, a missed frame range).

Both compare the output to the working input only inside the detected mask, so the
checks are cheap and do not need a clean reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from wm_platform.watermark_detect import DetectionResult, static_overlay_mask

UNCHANGED_MEAN_ABS_DIFF = 1.5  # grey levels inside the mask; codec noise alone stays below
RESIDUAL_OVERLAP = 0.35  # share of the original static mask the detector still finds
SAMPLE_FRAMES = 12


@dataclass(frozen=True)
class QualityVerdict:
    status: str  # pass | unchanged | residual_suspected | skipped
    details: dict[str, object]

    @property
    def unchanged(self) -> bool:
        return self.status == "unchanged"

    @property
    def residual(self) -> bool:
        return self.status == "residual_suspected"

    def as_report(self) -> dict[str, object]:
        return {"status": self.status, **self.details}


def check_output(input_path: Path, output_path: Path, detection: DetectionResult) -> QualityVerdict:
    mask = detection.union_mask
    if mask is None or not mask.any():
        return QualityVerdict("skipped", {"reason": "no detection mask"})
    diff = masked_difference(input_path, output_path, mask)
    if diff is None:
        return QualityVerdict("skipped", {"reason": "could not read frames"})
    details: dict[str, object] = {"mask_mean_abs_diff": round(diff, 3)}
    if diff < UNCHANGED_MEAN_ABS_DIFF:
        return QualityVerdict("unchanged", details)
    if detection.temporal:
        # Moving overlays are not static by definition; the static re-detector cannot
        # judge them. The unchanged check above still applies.
        details["residual_check"] = "skipped_moving_overlay"
        return QualityVerdict("pass", details)
    residual = static_overlay_mask(output_path)
    if residual is None:
        details["residual_check"] = "unreadable"
        return QualityVerdict("pass", details)
    if residual.shape != mask.shape:
        residual = cv2.resize(residual.astype(np.uint8), (mask.shape[1], mask.shape[0])) > 0
    overlap = float((residual & mask).sum()) / float(mask.sum())
    details["residual_overlap"] = round(overlap, 4)
    if overlap >= RESIDUAL_OVERLAP:
        return QualityVerdict("residual_suspected", details)
    return QualityVerdict("pass", details)


def masked_difference(input_path: Path, output_path: Path, mask: np.ndarray) -> float | None:
    a = _sample(input_path)
    b = _sample(output_path)
    count = min(len(a), len(b))
    if count == 0:
        return None
    values = []
    for index in range(count):
        src = a[index]
        out = b[index]
        if out.shape[:2] != src.shape[:2]:
            out = cv2.resize(out, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_AREA)
        region = mask
        if region.shape != src.shape[:2]:
            region = cv2.resize(mask.astype(np.uint8), (src.shape[1], src.shape[0])) > 0
        values.append(float(np.abs(src.astype(np.int16) - out.astype(np.int16))[region].mean()))
    return float(np.mean(values))


def _sample(path: Path, count: int = SAMPLE_FRAMES) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return []
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frames: list[np.ndarray] = []
        indexes = np.linspace(0, max(total - 1, 0), num=min(count, total), dtype=int) if total > 0 else []
        for index in indexes:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        return frames
    finally:
        capture.release()
