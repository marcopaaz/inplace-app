from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scenedetect import ContentDetector, detect

from inplace.errors import PipelineError
from inplace.jobs import JobStore, hash_file
from inplace.models import SignalSample, SignalScan
from inplace.runner import run_cached

SIGNALS_VERSION = "1"


def median_flow(frames: list[np.ndarray]) -> float:
    if len(frames) < 2:
        return 0.0
    magnitudes: list[float] = []
    previous = _prepare(frames[0])
    for frame in frames[1:]:
        current = _prepare(frame)
        magnitudes.append(_flow(previous, current))
        previous = current
    return float(np.median(magnitudes))


def cut_times(path: Path) -> list[float]:
    scenes = detect(str(path), ContentDetector(), show_progress=False, backend="opencv")
    return [float(start.seconds) for start, _end in scenes[1:]]


def scan_signals(path: Path) -> SignalScan:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise PipelineError(f"Could not open {path.name}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    step = max(1, int(round(fps / 4)))
    samples: list[SignalSample] = []
    previous: np.ndarray | None = None
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            gray = _prepare(frame)
            flow = 0.0 if previous is None else _flow(previous, gray)
            samples.append(SignalSample(time_s=index / fps, flow=flow, luminance=_center_luminance(gray)))
            previous = gray
        index += 1
    capture.release()
    return SignalScan(fps=fps, samples=samples, cut_times=cut_times(path))


def run_signals(job: JobStore, force: set[str]) -> str:
    source = job.path("source.mp4")

    def produce() -> None:
        job.write_model("signals.json", scan_signals(source))

    return run_cached(
        job,
        stage="signals",
        version=SIGNALS_VERSION,
        inputs={"source": hash_file(source)},
        outputs=["signals.json"],
        force="signals" in force,
        fn=produce,
    )


def _prepare(frame: np.ndarray) -> np.ndarray:
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if width <= 320:
        return gray
    scaled_h = max(2, int(round(height * 320 / width)))
    return cv2.resize(gray, (320, scaled_h), interpolation=cv2.INTER_AREA)


def _flow(previous: np.ndarray, current: np.ndarray) -> float:
    flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(np.median(magnitude))


def _center_luminance(gray: np.ndarray) -> float:
    height, width = gray.shape[:2]
    crop = gray[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4]
    return float(crop.mean()) if crop.size else float(gray.mean())
