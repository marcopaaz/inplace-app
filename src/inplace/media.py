from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from inplace.errors import PipelineError


def ffprobe(path: Path) -> dict[str, object]:
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode(errors="replace")[-400:]
        raise PipelineError(f"ffprobe failed for {path.name}: {detail}") from exc
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise PipelineError(f"ffprobe returned an unexpected payload for {path.name}")
    return payload


def parse_fps(value: str) -> float:
    numerator, separator, denominator = value.partition("/")
    if separator:
        den = float(denominator)
        return float(numerator) / den if den else 0.0
    return float(value)


def extract_window(source: Path, dest: Path, start_s: float, end_s: float) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(source),
                "-ss",
                f"{start_s:.3f}",
                "-to",
                f"{end_s:.3f}",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(dest),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode(errors="replace")[-400:]
        raise PipelineError(f"Could not cut the window: {detail}") from exc


def count_frames(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise PipelineError(f"Could not open {path.name}")
    count = 0
    while True:
        ok, _frame = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    if count < 1:
        raise PipelineError(f"No frames in {path.name}")
    return count


def read_frames(path: Path, stride: int = 1) -> tuple[list[np.ndarray], float]:
    if stride < 1:
        raise PipelineError("Frame stride must be at least 1")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise PipelineError(f"Could not open {path.name}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    frames: list[np.ndarray] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            frames.append(frame)
        index += 1
    capture.release()
    if not frames:
        raise PipelineError(f"No frames in {path.name}")
    return frames, fps / stride


def frame_at(path: Path, time_s: float) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise PipelineError(f"Could not open {path.name}")
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_s) * 1000.0)
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise PipelineError(f"Could not read a frame at {time_s:.2f}s")
    return frame


def write_mp4(path: Path, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        raise PipelineError("Refusing to write a video with no frames")
    height, width = frames[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in frames:
            if frame.shape[0] != height or frame.shape[1] != width:
                raise PipelineError("Frames changed size during encode")
            process.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        process.stdin.close()
        stderr = process.stderr.read() if process.stderr is not None else b""
        code = process.wait()
    if code != 0:
        detail = stderr.decode(errors="replace")[-400:]
        raise PipelineError(f"ffmpeg encode failed: {detail}")


def resize_width(frame: np.ndarray, width: int) -> np.ndarray:
    height, current = frame.shape[:2]
    if current <= width:
        return frame
    scaled_h = max(2, int(round(height * width / current)))
    scaled_h -= scaled_h % 2
    return cv2.resize(frame, (width, scaled_h), interpolation=cv2.INTER_AREA)
