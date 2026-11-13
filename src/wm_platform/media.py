"""FFmpeg helpers: probe, ingest normalization, chunk planning and final assembly.

Every stage after ingest (OpenCV detector, VideoHelperSuite loader, MiniMax/DiffuEraser)
indexes frames by position. That only lines up when the working copy has a constant frame
rate, no rotation side data and even dimensions, so inputs that break any of those are
re-encoded once into a canonical intermediate before anything else touches them.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from wm_platform.errors import AppError

# Intermediate/final encodes: visually lossless, one generation only.
INTERMEDIATE_CRF = "12"
FINAL_CRF = "16"
VFR_TOLERANCE = 0.01


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float | None
    has_audio: bool
    pix_fmt: str | None = None
    rotation: int = 0
    variable_frame_rate: bool = False
    codec: str | None = None

    def as_report(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 3),
            "frame_count": self.frame_count,
            "duration": round(self.duration, 3) if self.duration else None,
            "has_audio": self.has_audio,
            "codec": self.codec,
            "pix_fmt": self.pix_fmt,
            "rotation": self.rotation,
            "variable_frame_rate": self.variable_frame_rate,
        }


@dataclass(frozen=True)
class Chunk:
    """A window of source frames sent to the inpainter in one call.

    `keep_from` is how many leading frames of this window's output are dropped at
    assembly because the previous window already produced them (the last window is
    slid back so every window keeps the model's preferred length).
    """

    start: int
    count: int
    keep_from: int = 0


@dataclass
class NormalizedInput:
    path: Path
    info: VideoInfo
    source_info: VideoInfo
    reasons: list[str] = field(default_factory=list)

    @property
    def normalized(self) -> bool:
        return bool(self.reasons)


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise AppError("PROVIDER_NOT_AVAILABLE", f"{tool} not found in PATH", 503)
    return path


def _run(command: list[str], what: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip().splitlines()
        detail = stderr[-1] if stderr else f"{what} failed"
        raise AppError("MEDIA_PROCESSING_FAILED", f"{what}: {detail}", 500)
    return completed


def _parse_rate(raw: object) -> float:
    try:
        value = Fraction(str(raw))
    except (ValueError, ZeroDivisionError):
        return 0.0
    return float(value) if value > 0 else 0.0


def _rotation(stream: dict[str, object]) -> int:
    tags = stream.get("tags")
    if isinstance(tags, dict) and str(tags.get("rotate", "")).lstrip("-").isdigit():
        return int(tags["rotate"]) % 360
    for side in stream.get("side_data_list") or []:
        if isinstance(side, dict) and "rotation" in side:
            try:
                return int(float(side["rotation"])) % 360
            except (TypeError, ValueError):
                return 0
    return 0


def probe_video(path: Path) -> VideoInfo:
    ffprobe = _require("ffprobe")
    completed = _run(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        "ffprobe",
    )
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise AppError("MEDIA_PROCESSING_FAILED", f"ffprobe returned invalid json: {exc.msg}", 500) from exc
    streams = payload.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise AppError("MEDIA_TYPE_NOT_SUPPORTED", "input has no video stream", 400)
    fmt = payload.get("format") or {}
    r_rate = _parse_rate(video.get("r_frame_rate"))
    avg_rate = _parse_rate(video.get("avg_frame_rate"))
    fps = avg_rate or r_rate or 24.0
    duration = _as_float(video.get("duration")) or _as_float(fmt.get("duration"))
    frame_count = _as_int(video.get("nb_frames"))
    if not frame_count:
        frame_count = count_frames(path) if duration is None else int(round(duration * fps))
    vfr = bool(r_rate and avg_rate and abs(r_rate - avg_rate) / r_rate > VFR_TOLERANCE)
    return VideoInfo(
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=fps,
        frame_count=int(frame_count or 0),
        duration=duration,
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        pix_fmt=video.get("pix_fmt"),
        rotation=_rotation(video),
        variable_frame_rate=vfr,
        codec=video.get("codec_name"),
    )


def count_frames(path: Path) -> int:
    ffprobe = _require("ffprobe")
    completed = _run(
        [
            ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets",
            "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path),
        ],
        "ffprobe frame count",
    )
    return _as_int((completed.stdout or "").strip().split(",")[0]) or 0


def probe_video_duration_seconds(path: Path) -> float | None:
    try:
        info = probe_video(path)
    except AppError:
        return None
    return info.duration if info.duration and info.duration > 0 else None


def normalization_reasons(info: VideoInfo) -> list[str]:
    reasons: list[str] = []
    if info.variable_frame_rate:
        reasons.append("variable_frame_rate")
    if info.rotation:
        reasons.append(f"rotation_{info.rotation}")
    if info.width % 2 or info.height % 2:
        reasons.append("odd_dimensions")
    if info.pix_fmt and info.pix_fmt not in {"yuv420p", "yuvj420p"}:
        reasons.append(f"pix_fmt_{info.pix_fmt}")
    return reasons


def normalize_input(source: Path, work_dir: Path) -> NormalizedInput:
    """Return a working copy every later stage can index by frame number."""
    source_info = probe_video(source)
    reasons = normalization_reasons(source_info)
    if not reasons:
        return NormalizedInput(path=source, info=source_info, source_info=source_info)
    ffmpeg = _require("ffmpeg")
    work_dir.mkdir(parents=True, exist_ok=True)
    target = work_dir / "normalized.mkv"
    fps = Fraction(source_info.fps).limit_denominator(1001)
    # ffmpeg applies rotation side data by default (autorotate); the filter chain pins
    # a constant frame rate and even dimensions.
    _run(
        [
            ffmpeg, "-y", "-v", "error", "-i", str(source),
            "-map", "0:v:0", "-map", "0:a?",
            "-vf", f"fps={fps.numerator}/{fps.denominator},scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1",
            "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "medium", "-crf", INTERMEDIATE_CRF,
            "-c:a", "copy", "-metadata:s:v:0", "rotate=0",
            str(target),
        ],
        "normalize input",
    )
    return NormalizedInput(path=target, info=probe_video(target), source_info=source_info, reasons=reasons)


def plan_chunks(frame_count: int, chunk_frames: int, *, frame_multiple: int = 4) -> list[Chunk]:
    """Split a clip into windows of `frame_multiple * k + 1` frames.

    Wan-based removers (MiniMax-Remover) compress time 4x, so only 4k+1 frame windows
    come back complete; anything else is padded by the node with frozen frames. Windows
    never exceed `chunk_frames`. A remainder is handled by sliding the last window back
    to end on the final frame and dropping the overlap at assembly. `chunk_frames <= 0`
    means one window for the whole clip.
    """
    if frame_count <= 0:
        return []

    def fit(n: int) -> int:
        if frame_multiple <= 1 or n < frame_multiple + 1:
            return n
        return ((n - 1) // frame_multiple) * frame_multiple + 1

    window = fit(frame_count if chunk_frames <= 0 else min(chunk_frames, frame_count))
    if window <= 0:
        return []
    chunks: list[Chunk] = []
    start = 0
    while start + window <= frame_count:
        chunks.append(Chunk(start=start, count=window))
        start += window
    if start < frame_count:
        tail_start = frame_count - window
        chunks.append(Chunk(start=tail_start, count=window, keep_from=start - tail_start))
    return chunks


def assemble_output(
    segments: list[tuple[Path, Chunk]],
    audio_source: Path | None,
    destination: Path,
    *,
    fps: float,
) -> Path:
    """Concatenate inpainted windows (dropping overlaps) and mux the source audio back."""
    if not segments:
        raise AppError("MEDIA_PROCESSING_FAILED", "no inpainted segments to assemble", 500)
    ffmpeg = _require("ffmpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, "-y", "-v", "error"]
    for path, _chunk in segments:
        command += ["-i", str(path)]
    audio_index = len(segments)
    if audio_source is not None:
        command += ["-i", str(audio_source)]
    rate = Fraction(fps).limit_denominator(1001)
    parts = []
    for index, (_path, chunk) in enumerate(segments):
        parts.append(
            f"[{index}:v:0]trim=start_frame={chunk.keep_from},setpts=PTS-STARTPTS,"
            f"fps={rate.numerator}/{rate.denominator},format=yuv420p[v{index}]"
        )
    joined = "".join(f"[v{index}]" for index in range(len(segments)))
    graph = ";".join(parts) + f";{joined}concat=n={len(segments)}:v=1:a=0[vout]"
    command += ["-filter_complex", graph, "-map", "[vout]"]
    if audio_source is not None:
        command += ["-map", f"{audio_index}:a?", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    command += ["-c:v", "libx264", "-preset", "medium", "-crf", FINAL_CRF, "-movflags", "+faststart"]
    temp = destination.with_name(f"{destination.stem}.tmp{destination.suffix}")
    command.append(str(temp))
    try:
        _run(command, "assemble output")
        temp.replace(destination)
    finally:
        if temp.exists():
            temp.unlink()
    return destination


def _as_float(value: object) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _as_int(value: object) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None
