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
DEFAULT_PROBE_TIMEOUT_SECONDS = 60.0
DEFAULT_MEDIA_TIMEOUT_SECONDS = 300.0

CONTAINER_COPY_AUDIO_CODECS: dict[str, frozenset[str]] = {
    ".mp4": frozenset({"aac", "mp3", "ac3", "eac3", "alac"}),
    ".mov": frozenset(
        {"aac", "mp3", "ac3", "eac3", "alac", "pcm_s16le", "pcm_s16be", "pcm_s24le", "pcm_s24be", "pcm_f32le"}
    ),
    ".mkv": frozenset(
        {"aac", "mp3", "ac3", "eac3", "alac", "opus", "vorbis", "flac", "pcm_s16le", "pcm_s24le", "pcm_f32le"}
    ),
}

SDR_COLOR_SPACES = frozenset({"bt709", "bt470bg", "smpte170m", "smpte240m", "fcc", "gbr", "ycgco"})
SDR_COLOR_PRIMARIES = frozenset(
    {"bt709", "bt470m", "bt470bg", "smpte170m", "smpte240m", "film", "smpte431", "smpte432"}
)
SDR_COLOR_TRANSFERS = frozenset(
    {"bt709", "gamma22", "gamma28", "smpte170m", "smpte240m", "linear", "iec61966-2-1", "iec61966-2-4", "bt1361e"}
)
HDR_TRANSFERS = frozenset({"smpte2084", "arib-std-b67"})
HDR_PRIMARIES = frozenset({"bt2020"})
HDR_SPACES = frozenset({"bt2020nc", "bt2020c", "bt2020_ncl", "bt2020_cl"})


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
    audio_codec: str | None = None
    color_space: str | None = None
    color_primaries: str | None = None
    color_transfer: str | None = None
    color_range: str | None = None

    @property
    def is_hdr(self) -> bool:
        return (
            (self.color_transfer or "") in HDR_TRANSFERS
            or (self.color_primaries or "") in HDR_PRIMARIES
            or (self.color_space or "") in HDR_SPACES
        )

    def as_report(self) -> dict[str, object]:
        report: dict[str, object] = {
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
        if self.audio_codec:
            report["audio_codec"] = self.audio_codec
        if self.color_space:
            report["color_space"] = self.color_space
        if self.color_primaries:
            report["color_primaries"] = self.color_primaries
        if self.color_transfer:
            report["color_transfer"] = self.color_transfer
        return report


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


def _run(
    command: list[str],
    what: str,
    *,
    timeout: float | None = DEFAULT_MEDIA_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[str]:
    effective_timeout = timeout if (timeout is not None and timeout > 0) else None
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=effective_timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise AppError(
            "MEDIA_PROCESSING_FAILED",
            f"{what} timed out after {effective_timeout:.1f}s",
            504,
        ) from exc
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip().splitlines()
        detail = stderr[-1] if stderr else f"{what} failed"
        raise AppError("MEDIA_PROCESSING_FAILED", f"{what}: {detail}", 500)
    return completed


def _scaled_media_timeout(base_timeout: float | None, duration_seconds: float | None) -> float | None:
    if base_timeout is None:
        return None
    if base_timeout <= 0:
        return None
    extra = max(0.0, float(duration_seconds or 0.0) * 4.0)
    return max(float(base_timeout), 60.0 + extra)


def _clean_tag(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip().lower()
    if not cleaned or cleaned in {"unknown", "unspecified", "reserved"}:
        return None
    return cleaned


def sdr_color_args(info: VideoInfo | None) -> list[str]:
    """FFmpeg output stream flags preserving SDR VUI/container color metadata."""
    if info is None or info.is_hdr:
        return []
    args: list[str] = []
    if info.color_space in SDR_COLOR_SPACES:
        args += ["-colorspace", info.color_space]
    if info.color_primaries in SDR_COLOR_PRIMARIES:
        args += ["-color_primaries", info.color_primaries]
    if info.color_transfer in SDR_COLOR_TRANSFERS:
        args += ["-color_trc", info.color_transfer]
    return args


def sdr_setparams_filter(info: VideoInfo | None) -> str:
    """Filter-level frame color tagger so FFmpeg 8+/9+ filtergraphs keep SDR metadata."""
    if info is None or info.is_hdr:
        return ""
    params: list[str] = []
    if info.color_space in SDR_COLOR_SPACES:
        params.append(f"colorspace={info.color_space}")
    if info.color_primaries in SDR_COLOR_PRIMARIES:
        params.append(f"color_primaries={info.color_primaries}")
    if info.color_transfer in SDR_COLOR_TRANSFERS:
        params.append(f"color_trc={info.color_transfer}")
    return f"setparams={':'.join(params)}" if params else ""


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


def probe_video(path: Path, *, timeout: float | None = DEFAULT_PROBE_TIMEOUT_SECONDS) -> VideoInfo:
    ffprobe = _require("ffprobe")
    completed = _run(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        "ffprobe",
        timeout=timeout,
    )
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise AppError("MEDIA_PROCESSING_FAILED", f"ffprobe returned invalid json: {exc.msg}", 500) from exc
    streams = payload.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise AppError("MEDIA_TYPE_NOT_SUPPORTED", "input has no video stream", 400)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fmt = payload.get("format") or {}
    r_rate = _parse_rate(video.get("r_frame_rate"))
    avg_rate = _parse_rate(video.get("avg_frame_rate"))
    fps = avg_rate or r_rate or 24.0
    duration = _as_float(video.get("duration")) or _as_float(fmt.get("duration"))
    frame_count = _as_int(video.get("nb_frames"))
    if not frame_count:
        frame_count = count_frames(path, timeout=timeout) if duration is None else int(round(duration * fps))
    vfr = bool(r_rate and avg_rate and abs(r_rate - avg_rate) / r_rate > VFR_TOLERANCE)
    return VideoInfo(
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=fps,
        frame_count=int(frame_count or 0),
        duration=duration,
        has_audio=audio is not None,
        pix_fmt=_clean_tag(video.get("pix_fmt")),
        rotation=_rotation(video),
        variable_frame_rate=vfr,
        codec=_clean_tag(video.get("codec_name")),
        audio_codec=_clean_tag(audio.get("codec_name")) if isinstance(audio, dict) else None,
        color_space=_clean_tag(video.get("color_space")),
        color_primaries=_clean_tag(video.get("color_primaries")),
        color_transfer=_clean_tag(video.get("color_transfer")),
        color_range=_clean_tag(video.get("color_range")),
    )


def probe_audio_codec(path: Path, *, timeout: float | None = DEFAULT_PROBE_TIMEOUT_SECONDS) -> str | None:
    ffprobe = _require("ffprobe")
    try:
        completed = _run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "csv=p=0",
                str(path),
            ],
            "ffprobe audio codec",
            timeout=timeout,
        )
    except AppError:
        return None
    line = (completed.stdout or "").strip().splitlines()
    return _clean_tag(line[0].split(",")[0]) if line else None


def count_frames(path: Path, *, timeout: float | None = DEFAULT_PROBE_TIMEOUT_SECONDS) -> int:
    ffprobe = _require("ffprobe")
    completed = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            str(path),
        ],
        "ffprobe frame count",
        timeout=timeout,
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


def normalize_input(
    source: Path,
    work_dir: Path,
    *,
    timeout: float | None = DEFAULT_MEDIA_TIMEOUT_SECONDS,
) -> NormalizedInput:
    """Return a working copy every later stage can index by frame number."""
    source_info = probe_video(source, timeout=timeout)
    reasons = normalization_reasons(source_info)
    if not reasons:
        return NormalizedInput(path=source, info=source_info, source_info=source_info)
    ffmpeg = _require("ffmpeg")
    work_dir.mkdir(parents=True, exist_ok=True)
    target = work_dir / "normalized.mkv"
    fps = Fraction(source_info.fps).limit_denominator(1001)
    vf_parts = [
        f"fps={fps.numerator}/{fps.denominator}",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "setsar=1",
        "format=yuv420p",
    ]
    setparams = sdr_setparams_filter(source_info)
    if setparams:
        vf_parts.append(setparams)
    # ffmpeg applies rotation side data by default (autorotate); the filter chain pins
    # a constant frame rate and even dimensions while preserving SDR color tags.
    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        ",".join(vf_parts),
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        INTERMEDIATE_CRF,
        *sdr_color_args(source_info),
        "-c:a",
        "copy",
        "-metadata:s:v:0",
        "rotate=0",
        str(target),
    ]
    _run(command, "normalize input", timeout=_scaled_media_timeout(timeout, source_info.duration))
    return NormalizedInput(path=target, info=probe_video(target, timeout=timeout), source_info=source_info, reasons=reasons)


def extract_frame_window(
    source: Path,
    destination: Path,
    *,
    start: int,
    count: int,
    fps: float,
    color_info: VideoInfo | None = None,
    timeout: float | None = DEFAULT_MEDIA_TIMEOUT_SECONDS,
) -> Path:
    """Extract an exact [start, start + count) frame slice without running ComfyUI."""
    if count <= 0:
        raise AppError("MEDIA_PROCESSING_FAILED", "frame window count must be positive", 500)
    ffmpeg = _require("ffmpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rate = Fraction(fps).limit_denominator(1001)
    vf_parts = [
        f"trim=start_frame={max(0, int(start))}:end_frame={max(0, int(start)) + int(count)}",
        "setpts=PTS-STARTPTS",
        f"fps={rate.numerator}/{rate.denominator}",
        "format=yuv420p",
    ]
    setparams = sdr_setparams_filter(color_info)
    if setparams:
        vf_parts.append(setparams)
    temp = destination.with_name(f"{destination.stem}.tmp{destination.suffix}")
    command = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-vf",
        ",".join(vf_parts),
        "-pix_fmt",
        "yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        INTERMEDIATE_CRF,
        *sdr_color_args(color_info),
        "-an",
        str(temp),
    ]
    duration = count / fps if fps > 0 else None
    try:
        _run(command, "extract clean frame window", timeout=_scaled_media_timeout(timeout, duration))
        temp.replace(destination)
    finally:
        if temp.exists():
            temp.unlink()
    return destination


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


def _resolve_color_info(
    color_info: VideoInfo | None,
    segments: list[tuple[Path, Chunk]],
    audio_source: Path | None,
    *,
    timeout: float | None,
) -> VideoInfo | None:
    if color_info is not None:
        return color_info
    probe_timeout = min(timeout, DEFAULT_PROBE_TIMEOUT_SECONDS) if timeout else DEFAULT_PROBE_TIMEOUT_SECONDS
    for candidate in ([audio_source] if audio_source is not None else []) + [segments[0][0]]:
        if candidate is None:
            continue
        try:
            info = probe_video(candidate, timeout=probe_timeout)
        except AppError:
            continue
        if info.color_space or info.color_primaries or info.color_transfer or info.is_hdr:
            return info
    return None


def _audio_mux_args(audio_source: Path, destination: Path, *, timeout: float | None) -> list[str]:
    probe_timeout = min(timeout, DEFAULT_PROBE_TIMEOUT_SECONDS) if timeout else DEFAULT_PROBE_TIMEOUT_SECONDS
    codec = probe_audio_codec(audio_source, timeout=probe_timeout)
    allowed = CONTAINER_COPY_AUDIO_CODECS.get(destination.suffix.lower(), frozenset())
    if codec and codec in allowed:
        return ["-c:a", "copy"]
    return ["-c:a", "aac", "-b:a", "192k"]


def assemble_output(
    segments: list[tuple[Path, Chunk]],
    audio_source: Path | None,
    destination: Path,
    *,
    fps: float,
    color_info: VideoInfo | None = None,
    timeout: float | None = DEFAULT_MEDIA_TIMEOUT_SECONDS,
) -> Path:
    """Concatenate inpainted windows (dropping overlaps) and mux the source audio back."""
    if not segments:
        raise AppError("MEDIA_PROCESSING_FAILED", "no inpainted segments to assemble", 500)
    ffmpeg = _require("ffmpeg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    resolved_color = _resolve_color_info(color_info, segments, audio_source, timeout=timeout)

    total_frames = 0
    for seg_path, chunk in segments:
        if chunk.count > 0:
            total_frames += max(0, chunk.count - chunk.keep_from)
        else:
            try:
                seg_info = probe_video(seg_path, timeout=DEFAULT_PROBE_TIMEOUT_SECONDS)
                total_frames += max(0, seg_info.frame_count - chunk.keep_from)
            except AppError:
                total_frames = 0
                break
    target_duration = (total_frames / fps) if (total_frames > 0 and fps > 0) else None

    command = [ffmpeg, "-y", "-v", "error"]
    for path, _chunk in segments:
        command += ["-i", str(path)]
    audio_index = len(segments)
    if audio_source is not None:
        if target_duration is not None:
            # Clamp overlong audio at input demux so video frames are never truncated by -shortest.
            command += ["-t", f"{target_duration:.6f}"]
        command += ["-i", str(audio_source)]
    rate = Fraction(fps).limit_denominator(1001)
    setparams = sdr_setparams_filter(resolved_color)
    parts = []
    for index, (_path, chunk) in enumerate(segments):
        chain = [
            f"trim=start_frame={chunk.keep_from}",
            "setpts=PTS-STARTPTS",
            f"fps={rate.numerator}/{rate.denominator}",
            "format=yuv420p",
        ]
        if setparams:
            chain.append(setparams)
        parts.append(f"[{index}:v:0]{','.join(chain)}[v{index}]")
    joined = "".join(f"[v{index}]" for index in range(len(segments)))
    graph = ";".join(parts) + f";{joined}concat=n={len(segments)}:v=1:a=0[vout]"
    command += ["-filter_complex", graph, "-map", "[vout]"]
    if audio_source is not None:
        command += ["-map", f"{audio_index}:a?", *_audio_mux_args(audio_source, destination, timeout=timeout)]
    command += [
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        FINAL_CRF,
        *sdr_color_args(resolved_color),
        "-movflags",
        "+faststart",
    ]
    temp = destination.with_name(f"{destination.stem}.tmp{destination.suffix}")
    command.append(str(temp))
    try:
        _run(command, "assemble output", timeout=_scaled_media_timeout(timeout, target_duration))
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
