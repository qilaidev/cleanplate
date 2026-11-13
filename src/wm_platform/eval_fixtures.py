"""Synthetic eval fixtures.

Every fixture is rendered twice: the watermarked clip under `<split>/<name>.mp4` (lossy
mp4v, like real inputs) and the clean plate under `clean/<name>.mkv` (lossless FFV1 so
the reference carries no codec noise). The background is a moving textured pattern so a
detector cannot pass simply because nothing in the frame changes. Clean plates let
`eval_metrics` score outputs without a person judging them.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from wm_platform.eval_runner import default_dataset_root

FIXTURE_WIDTH = 160
FIXTURE_HEIGHT = 120
FIXTURE_FPS = 8
FIXTURE_FRAMES = 8
CLEAN_DIRNAME = "clean"
CLEAN_SUFFIX = ".mkv"
REGION_SUFFIX = ".region.png"
LOSSLESS_FOURCC = "FFV1"

MANIFEST_FIELDS = (
    "id",
    "split",
    "filename",
    "clean_filename",
    "region_filename",
    "watermark_type",
    "position",
    "moving",
    "notes",
    "expected",
)

Draw = Callable[[int, np.ndarray], None]


def write_static_corner(
    path: Path,
    *,
    corner: str = "top_left",
    frames: int = FIXTURE_FRAMES,
    clean_path: Path | None = None,
) -> Path:
    if corner == "top_left":
        x, y = 4, 4
    else:
        x = FIXTURE_WIDTH - 48 - 4
        y = FIXTURE_HEIGHT - 28 - 4
    return _write_pair(
        path,
        clean_path,
        frames,
        lambda _index, frame: _paint_block(frame, x, y, 44, 24, (20, 180, 255)),
    )


def write_translucent_block(path: Path, *, frames: int = FIXTURE_FRAMES, clean_path: Path | None = None) -> Path:
    def _draw(_index: int, frame: np.ndarray) -> None:
        overlay = frame.copy()
        overlay[6:32, 6:56] = (245, 245, 245)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    return _write_pair(path, clean_path, frames, _draw)


def write_drifting_block(path: Path, *, frames: int = FIXTURE_FRAMES, clean_path: Path | None = None) -> Path:
    def _draw(index: int, frame: np.ndarray) -> None:
        x = 4 + index * 12
        _paint_block(frame, x, 6, 44, 22, (20, 180, 255))

    return _write_pair(path, clean_path, frames, _draw)


def write_large_block(path: Path, *, frames: int = FIXTURE_FRAMES, clean_path: Path | None = None) -> Path:
    def _draw(_index: int, frame: np.ndarray) -> None:
        _paint_grid(frame, 0, 0, FIXTURE_WIDTH, 52)

    return _write_pair(path, clean_path, frames, _draw)


def write_multi_block(path: Path, *, frames: int = FIXTURE_FRAMES, clean_path: Path | None = None) -> Path:
    def _draw(_index: int, frame: np.ndarray) -> None:
        boxes = (
            (2, 2),
            (56, 2),
            (110, 2),
            (2, 46),
            (110, 46),
            (2, 90),
            (56, 90),
            (110, 90),
        )
        for x, y in boxes:
            _paint_block(frame, x, y, 28, 18, (20, 180, 255))

    return _write_pair(path, clean_path, frames, _draw)


def write_center_overlap(path: Path, *, frames: int = FIXTURE_FRAMES, clean_path: Path | None = None) -> Path:
    def _draw(_index: int, frame: np.ndarray) -> None:
        _paint_grid(frame, 8, 8, FIXTURE_WIDTH - 16, FIXTURE_HEIGHT - 16)

    return _write_pair(path, clean_path, frames, _draw)


def fixture_specs() -> list[dict[str, str]]:
    return [
        {
            "id": "simple-tl",
            "split": "simple",
            "filename": "simple_top_left.mp4",
            "watermark_type": "static_logo",
            "position": "top_left",
            "moving": "false",
            "notes": "static opaque top-left block over a moving background",
            "expected": "detectable",
            "kind": "simple_tl",
        },
        {
            "id": "simple-br",
            "split": "simple",
            "filename": "simple_bottom_right.mp4",
            "watermark_type": "static_logo",
            "position": "bottom_right",
            "moving": "false",
            "notes": "static opaque bottom-right block over a moving background",
            "expected": "detectable",
            "kind": "simple_br",
        },
        {
            "id": "medium-translucent",
            "split": "medium",
            "filename": "medium_translucent.mp4",
            "watermark_type": "translucent_logo",
            "position": "top_left",
            "moving": "false",
            "notes": "semi-transparent corner block over a moving background; needs the persistent-edge path",
            "expected": "detectable",
            "kind": "medium_translucent",
        },
        {
            "id": "medium-drift",
            "split": "medium",
            "filename": "medium_drift.mp4",
            "watermark_type": "moving_logo",
            "position": "top_left",
            "moving": "true",
            "notes": "drifting block; static detector must stay low, SAM3 tracking should produce per-frame masks",
            "expected": "sam3_or_low",
            "kind": "medium_drift",
        },
        {
            "id": "hard-large",
            "split": "hard",
            "filename": "hard_large.mp4",
            "watermark_type": "large_overlay",
            "position": "top_band",
            "moving": "false",
            "notes": "large static band that should trip the area gate",
            "expected": "reject",
            "kind": "hard_large",
        },
        {
            "id": "hard-multi",
            "split": "hard",
            "filename": "hard_multi.mp4",
            "watermark_type": "multi_logo",
            "position": "edges",
            "moving": "false",
            "notes": "many static edge blocks that should trip the region gate",
            "expected": "reject",
            "kind": "hard_multi",
        },
        {
            "id": "hard-center",
            "split": "hard",
            "filename": "hard_center.mp4",
            "watermark_type": "center_overlap",
            "position": "center",
            "moving": "false",
            "notes": "large textured overlay covering center and margins",
            "expected": "reject",
            "kind": "hard_center",
        },
    ]


def generate_eval_fixtures(dataset_root: Path) -> list[Path]:
    written: list[Path] = []
    rows: list[dict[str, str]] = []
    for spec in fixture_specs():
        split_dir = dataset_root / spec["split"]
        path = split_dir / spec["filename"]
        clean_name = Path(spec["filename"]).with_suffix(CLEAN_SUFFIX).name
        region_name = Path(spec["filename"]).stem + REGION_SUFFIX
        clean_path = dataset_root / CLEAN_DIRNAME / clean_name
        _write_kind(path, spec["kind"], clean_path)
        written.append(path)
        row = {field: spec.get(field, "") for field in MANIFEST_FIELDS}
        row["clean_filename"] = f"{CLEAN_DIRNAME}/{clean_name}"
        row["region_filename"] = f"{CLEAN_DIRNAME}/{region_name}"
        rows.append(row)
    _write_manifest(dataset_root / "manifest.csv", rows)
    return written


def _write_kind(path: Path, kind: str, clean_path: Path | None = None) -> Path:
    if kind == "simple_tl":
        return write_static_corner(path, corner="top_left", clean_path=clean_path)
    if kind == "simple_br":
        return write_static_corner(path, corner="bottom_right", clean_path=clean_path)
    if kind == "medium_translucent":
        return write_translucent_block(path, clean_path=clean_path)
    if kind == "medium_drift":
        return write_drifting_block(path, clean_path=clean_path)
    if kind == "hard_large":
        return write_large_block(path, clean_path=clean_path)
    if kind == "hard_multi":
        return write_multi_block(path, clean_path=clean_path)
    if kind == "hard_center":
        return write_center_overlap(path, clean_path=clean_path)
    raise ValueError(f"unknown fixture kind: {kind}")


def _paint_grid(frame: np.ndarray, x: int, y: int, width: int, height: int, line: int = 3, step: int = 6) -> None:
    x2 = min(frame.shape[1], x + width)
    y2 = min(frame.shape[0], y + height)
    x = max(0, x)
    y = max(0, y)
    region = frame[y:y2, x:x2]
    region[:] = (20, 20, 20)
    height_px, width_px = region.shape[:2]
    for row in range(0, height_px, step):
        region[row : min(height_px, row + line), :] = (240, 240, 240)
    for col in range(0, width_px, step):
        region[:, col : min(width_px, col + line)] = (20, 180, 255)


def _paint_block(frame: np.ndarray, x: int, y: int, width: int, height: int, color: tuple[int, int, int]) -> None:
    x2 = min(frame.shape[1], x + width)
    y2 = min(frame.shape[0], y + height)
    x = max(0, x)
    y = max(0, y)
    frame[y:y2, x:x2] = (240, 240, 240)
    inner_x = min(x2, x + 2)
    inner_y = min(y2, y + 2)
    inner_x2 = max(inner_x, x2 - 2)
    inner_y2 = max(inner_y, y2 - 2)
    frame[inner_y:inner_y2, inner_x:inner_x2] = color


def background_frame(index: int, *, width: int = FIXTURE_WIDTH, height: int = FIXTURE_HEIGHT) -> np.ndarray:
    """Deterministic moving background: scrolling gradient, drifting stripes, a moving disc."""
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    shift = index * 9.0
    base = 70.0 + 50.0 * np.sin((xs + shift) / 17.0) * np.cos((ys - shift * 0.6) / 13.0)
    stripes = ((xs + ys + shift * 1.5) // 11) % 2
    tone = base + stripes * 28.0
    frame = np.stack(
        [
            np.clip(tone * 0.9, 0, 255),
            np.clip(tone, 0, 255),
            np.clip(255 - tone * 0.8, 0, 255),
        ],
        axis=2,
    ).astype(np.uint8)
    # No per-pixel noise: at 160x120 the mp4v encoder would smear it into artifacts
    # that dominate every metric. Texture comes from the stripes and the moving disc.
    cx = int(30 + (index * 14) % max(1, width - 60))
    cy = int(height * 0.55)
    cv2.circle(frame, (cx, cy), 16, (200, 90, 40), thickness=-1)
    return frame


def _write_pair(path: Path, clean_path: Path | None, frames: int, draw: Draw) -> Path:
    _write_frames(path, frames, draw)
    if clean_path is not None:
        _write_frames(clean_path, frames, None)
        region_path = clean_path.with_name(clean_path.stem + REGION_SUFFIX)
        cv2.imwrite(str(region_path), painted_region(frames, draw).astype(np.uint8) * 255)
    return path


def painted_region(frames: int, draw: Draw) -> np.ndarray:
    """Union over frames of every pixel the overlay touches (exact, no codec noise)."""
    region = np.zeros((FIXTURE_HEIGHT, FIXTURE_WIDTH), dtype=bool)
    for index in range(max(2, frames)):
        base = background_frame(index)
        painted = base.copy()
        draw(index, painted)
        region |= np.any(painted != base, axis=2)
    return region


def _write_frames(path: Path, frames: int, draw: Draw | None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = LOSSLESS_FOURCC if path.suffix.lower() == CLEAN_SUFFIX else "mp4v"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*fourcc),
        FIXTURE_FPS,
        (FIXTURE_WIDTH, FIXTURE_HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to write fixture video: {path}")
    try:
        for index in range(max(2, frames)):
            frame = background_frame(index)
            if draw is not None:
                draw(index, frame)
            writer.write(frame)
    finally:
        writer.release()
    return path


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the in-repo eval fixtures and clean plates")
    parser.add_argument("--dataset-root", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    dataset_root = (args.dataset_root or default_dataset_root()).expanduser().resolve()
    written = generate_eval_fixtures(dataset_root)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
