from __future__ import annotations

import sys
from pathlib import Path

from inplace.constants import (
    MAX_RENDER_S,
    MIN_SURFACE_FRACTION,
    MIN_WINDOW_S,
    OCCLUDER_PROMPT,
    SAM_MODEL,
    TRACK_FPS,
)
from inplace.errors import PipelineError
from inplace.jobs import JobStore, hash_file
from inplace.mask_archive import save_mask_archive
from inplace.media import count_frames, extract_window, read_frames, write_mp4
from inplace.meta_client import MetaClient
from inplace.models import Placement
from inplace.runner import run_cached
from inplace.sam_masks import rasterize_sam
from inplace.stages.probe import probe_video

TRACK_VERSION = "2"


def run_track(job: JobStore, client: MetaClient, force: set[str]) -> str:
    placement = Placement.model_validate_json(job.path("placement.json").read_text())
    source = job.path("source.mp4")
    window_path = job.path("track/window.mp4")

    def produce() -> None:
        full = _sam_clip(job, source, window_path, placement.start_s, placement.end_s)
        window = probe_video(full)
        frame_count = count_frames(full)
        stride = _stride(window.fps)
        surface_text = _stream(job, client, full, placement.sam_prompt, placement.start_s, placement.end_s)
        person_text = _stream(job, client, full, OCCLUDER_PROMPT, placement.start_s, placement.end_s)
        kept = (frame_count + stride - 1) // stride
        print(f"tracking: reading {kept} surface masks", file=sys.stderr, flush=True)
        surface = rasterize_sam(
            surface_text,
            frame_count=frame_count,
            width=window.width,
            height=window.height,
            stride=stride,
        )
        print(f"tracking: reading {kept} occluder masks", file=sys.stderr, flush=True)
        person = rasterize_sam(
            person_text,
            frame_count=frame_count,
            width=window.width,
            height=window.height,
            stride=stride,
        )
        frames, fps = read_frames(full, stride)
        if len(frames) != len(surface):
            raise PipelineError("The tracked frames and the surface masks do not line up")
        write_mp4(window_path, frames, fps)
        visible_frames = sum(1 for mask in surface if float((mask > 0).mean()) >= MIN_SURFACE_FRACTION)
        visible_s = visible_frames / fps if fps else 0.0
        if visible_s < MIN_WINDOW_S:
            raise PipelineError(f"The surface is visible for only {visible_s:.1f}s")
        save_mask_archive(job.path("track/surface.zip"), surface, fps=fps)
        save_mask_archive(job.path("track/person.zip"), person, fps=fps)

    return run_cached(
        job,
        stage="track",
        version=TRACK_VERSION,
        inputs={
            "source": hash_file(source),
            "start": f"{placement.start_s:.3f}",
            "end": f"{placement.end_s:.3f}",
            "prompt": placement.sam_prompt,
            "occluder": OCCLUDER_PROMPT,
            "model": SAM_MODEL,
        },
        outputs=["track/window.mp4", "track/surface.zip", "track/person.zip"],
        force="track" in force,
        fn=produce,
    )


def render_bounds(start_s: float, end_s: float) -> tuple[float, float]:
    return start_s, min(end_s, start_s + MAX_RENDER_S)


def _stride(fps: float) -> int:
    if fps <= TRACK_FPS:
        return 1
    return max(1, int(round(fps / TRACK_FPS)))


def _sam_clip(job: JobStore, source: Path, window_path: Path, start_s: float, end_s: float) -> Path:
    """Return the clip whose frames match the saved SAM stream.

    A retry must not re-encode that clip. The mask frame numbers belong to the
    file that was uploaded, and a second encode can shift them.
    """
    full = job.path("track/full.mp4")
    marker = job.path("track/span.txt")
    span = f"{start_s:.3f}-{end_s:.3f}"
    if full.is_file() and marker.is_file() and marker.read_text() == span:
        return full
    legacy = (
        not full.is_file()
        and not marker.is_file()
        and window_path.is_file()
        and window_path.stat().st_size > 0
    )
    full.parent.mkdir(parents=True, exist_ok=True)
    if legacy:
        window_path.replace(full)
    else:
        extract_window(source, full, start_s, end_s)
    marker.write_text(span)
    return full


def _stream(
    job: JobStore,
    client: MetaClient,
    window_path: Path,
    prompt: str,
    start_s: float,
    end_s: float,
) -> str:
    slug = _slug(prompt)
    raw_path = job.path(f"paid/sam/{slug}-{start_s:.2f}-{end_s:.2f}.txt")
    if raw_path.is_file():
        return raw_path.read_text()
    file_id = client.upload(window_path)
    text = client.segment_video(file_id, prompt)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(text)
    return text


def _slug(text: str) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in text).strip("-")
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned[:80] or "prompt"
