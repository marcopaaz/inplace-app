from __future__ import annotations

from pathlib import Path

import numpy as np

from inplace.constants import OCCLUDER_PROMPT, SAM_MODEL
from inplace.errors import ModelError, PipelineError
from inplace.jobs import JobStore, hash_file
from inplace.media import frame_at, resize_width
from inplace.meta_client import MetaClient
from inplace.models import ScenePlan, SignalScan, SpotMeasurement, SpotReport, Video
from inplace.runner import run_cached
from inplace.sam_masks import rasterize_sam
from inplace.stages.score import placement_spans

SPOT_VERSION = "2"
SPOT_FRAMES = 2
SPOT_WIDTH = 640


def run_spot(job: JobStore, client: MetaClient, force: set[str]) -> str:
    source = job.path("source.mp4")
    plan = ScenePlan.model_validate_json(job.path("scene_plan.json").read_text())
    scan = SignalScan.model_validate_json(job.path("signals.json").read_text())
    video = Video.model_validate_json(job.path("video.json").read_text())

    def produce() -> None:
        measurements: list[SpotMeasurement] = []
        for start_s, end_s, _label, prompt in placement_spans(plan, scan, video.duration_s):
            measurements.append(_measure(job, client, source, start_s, end_s, prompt))
        job.write_model("spot.json", SpotReport(measurements=measurements))

    return run_cached(
        job,
        stage="spot",
        version=SPOT_VERSION,
        inputs={
            "source": hash_file(source),
            "plan": hash_file(job.path("scene_plan.json")),
            "signals": hash_file(job.path("signals.json")),
            "model": SAM_MODEL,
        },
        outputs=["spot.json"],
        force="spot" in force,
        fn=produce,
    )


def _measure(
    job: JobStore,
    client: MetaClient,
    source: Path,
    start_s: float,
    end_s: float,
    prompt: str,
) -> SpotMeasurement:
    times = _sample_times(start_s, end_s, SPOT_FRAMES)
    occlusions: list[float] = []
    best_area = 0
    aspect_w: int | None = None
    aspect_h: int | None = None
    last_error: ModelError | None = None
    answered = 0
    for time_s in times:
        frame = resize_width(frame_at(source, time_s), SPOT_WIDTH)
        height, width = frame.shape[:2]
        try:
            surface = _mask(job, client, frame, prompt, width, height, time_s)
        except ModelError as exc:
            last_error = exc
            continue
        answered += 1
        try:
            occluder = _mask(job, client, frame, OCCLUDER_PROMPT, width, height, time_s)
        except ModelError:
            occluder = np.zeros((height, width), dtype=np.uint8)
        area = int((surface > 0).sum())
        if area == 0:
            continue
        overlap = int(((surface > 0) & (occluder > 0)).sum())
        occlusions.append(overlap / area)
        if area > best_area:
            best_area = area
            ys, xs = np.where(surface > 0)
            aspect_w = int(xs.max() - xs.min() + 1)
            aspect_h = int(ys.max() - ys.min() + 1)
    if answered == 0 and last_error is not None:
        raise last_error
    return SpotMeasurement(
        start_s=round(start_s, 3),
        end_s=round(end_s, 3),
        sam_prompt=prompt,
        occlusion=None if not occlusions else float(sum(occlusions) / len(occlusions)),
        aspect_w=aspect_w,
        aspect_h=aspect_h,
        surface_found=best_area > 0,
    )


def _mask(
    job: JobStore,
    client: MetaClient,
    frame: np.ndarray,
    prompt: str,
    width: int,
    height: int,
    time_s: float,
) -> np.ndarray:
    text = _cached_segmentation(job, client, frame, prompt, time_s)
    if not text.strip():
        return np.zeros((height, width), dtype=np.uint8)
    last_error: PipelineError | None = None
    for media in ("video", "image"):
        try:
            return rasterize_sam(text, frame_count=1, width=width, height=height, media=media)[0]
        except PipelineError as exc:
            last_error = exc
    if last_error is None:
        raise PipelineError("SAM spot-check could not be read")
    raise last_error


def _cached_segmentation(
    job: JobStore,
    client: MetaClient,
    frame: np.ndarray,
    prompt: str,
    time_s: float,
) -> str:
    path = job.path(f"paid/spot/{_slug(prompt)}-{time_s:.2f}.txt")
    if path.is_file():
        return path.read_text()
    text = client.segment_image(frame, prompt)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text


def _slug(text: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in text).strip("-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned[:80] or "prompt"


def _sample_times(start_s: float, end_s: float, count: int) -> list[float]:
    span = end_s - start_s
    if span <= 0:
        return [start_s]
    if count <= 1:
        return [start_s + span / 2]
    return [start_s + span * (index + 0.5) / count for index in range(count)]
