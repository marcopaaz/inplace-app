from __future__ import annotations

from pathlib import Path

from inplace.constants import (
    ALLOWED_CODECS,
    MAX_LONG_EDGE,
    MAX_SHORT_EDGE,
    MAX_UPLOAD_S,
)
from inplace.errors import VideoRejected
from inplace.jobs import JobStore, hash_file
from inplace.media import ffprobe, parse_fps
from inplace.models import Video
from inplace.runner import run_cached

PROBE_VERSION = "1"


def rejection_reason(
    *,
    suffix: str,
    duration_s: float,
    width: int,
    height: int,
    codec: str,
) -> str | None:
    if suffix.lower() != ".mp4":
        return "Only MP4 uploads are accepted"
    if duration_s > MAX_UPLOAD_S:
        return f"Clip is {duration_s:.1f}s. The maximum is 3 minutes"
    long_edge = max(width, height)
    short_edge = min(width, height)
    if long_edge > MAX_LONG_EDGE or short_edge > MAX_SHORT_EDGE:
        return f"Resolution is {width}x{height}. The maximum is 1080p"
    if codec not in ALLOWED_CODECS:
        return f"Codec {codec} is not supported. Use H.264, H.265, or MPEG-4"
    return None


def probe_video(path: Path) -> Video:
    payload = ffprobe(path)
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise VideoRejected(f"{path.name} has no video stream")
    stream = next((item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video"), None)
    if not isinstance(stream, dict):
        raise VideoRejected(f"{path.name} has no video stream")
    coded_width = int(stream["width"])
    coded_height = int(stream["height"])
    rotation = _rotation(stream)
    width, height = coded_width, coded_height
    if rotation in {90, 270}:
        width, height = coded_height, coded_width
    codec = str(stream.get("codec_name") or "")
    fmt = payload.get("format")
    duration_s = 0.0
    if isinstance(fmt, dict) and fmt.get("duration") is not None:
        duration_s = float(str(fmt["duration"]))
    elif stream.get("duration") is not None:
        duration_s = float(str(stream["duration"]))
    rate = str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1")
    fps = parse_fps(rate)
    reason = rejection_reason(
        suffix=path.suffix,
        duration_s=duration_s,
        width=width,
        height=height,
        codec=codec,
    )
    if reason is not None:
        raise VideoRejected(reason)
    return Video(
        id=path.stem,
        file_path=str(path),
        duration_s=duration_s,
        fps=fps,
        width=coded_width,
        height=coded_height,
        codec=codec,
    )


def run_probe(job: JobStore, force: set[str]) -> str:
    source = job.path("source.mp4")

    def produce() -> None:
        video = probe_video(source)
        video.id = job.id
        video.file_path = "source.mp4"
        job.write_model("video.json", video)

    return run_cached(
        job,
        stage="probe",
        version=PROBE_VERSION,
        inputs={"source": hash_file(source)},
        outputs=["video.json"],
        force="probe" in force,
        fn=produce,
    )


def _rotation(stream: dict[str, object]) -> int:
    tags = stream.get("tags")
    if isinstance(tags, dict) and tags.get("rotate") is not None:
        try:
            return int(str(tags["rotate"])) % 360
        except ValueError:
            return 0
    side_data = stream.get("side_data_list")
    if isinstance(side_data, list):
        for item in side_data:
            if isinstance(item, dict) and item.get("rotation") is not None:
                try:
                    return int(str(item["rotation"])) % 360
                except ValueError:
                    return 0
    return 0
