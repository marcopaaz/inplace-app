from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from inplace.constants import JITTER_BUDGET, MAX_BAD_FRAME_FRACTION
from inplace.errors import PipelineError
from inplace.jobs import JobStore, hash_file
from inplace.mask_archive import MaskArchive
from inplace.media import frame_at, read_frames, write_mp4
from inplace.models import Placement, RenderRecord, ScenePlan, SignalScan, SpotReport, Video
from inplace.runner import run_cached
from inplace.stages.composite import composite_frame
from inplace.stages.report import visibility_over_frames
from inplace.stages.score import assemble_windows, score_window, summarize_windows
from inplace.stages.warp import estimate_homographies

WARP_VERSION = "2"
COMPOSITE_VERSION = "2"
REPORT_VERSION = "1"
SCORE_VERSION = "1"


def run_score(job: JobStore, force: set[str]) -> str:
    video = Video.model_validate_json(job.path("video.json").read_text())
    plan = ScenePlan.model_validate_json(job.path("scene_plan.json").read_text())
    scan = SignalScan.model_validate_json(job.path("signals.json").read_text())
    spots = SpotReport.model_validate_json(job.path("spot.json").read_text())

    def produce() -> None:
        thumbs = job.path("thumbs")
        if thumbs.exists():
            shutil.rmtree(thumbs)
        scored = [score_window(window) for window in assemble_windows(plan, scan, spots, video.duration_s)]
        report = summarize_windows(job.id, scored)
        source = job.path("source.mp4")
        for index, window in enumerate(report.windows):
            midpoint = (window.start_s + window.end_s) / 2
            frame = _thumbnail_frame(source, midpoint)
            relative = f"thumbs/{index}.jpg"
            target = job.path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(target), frame):
                raise PipelineError(f"Could not write the thumbnail for window {index}")
            window.thumbnail = relative
        job.write_model("placeability.json", report)

    outputs = ["placeability.json"]
    return run_cached(
        job,
        stage="score",
        version=SCORE_VERSION,
        inputs={
            "signals": hash_file(job.path("signals.json")),
            "plan": hash_file(job.path("scene_plan.json")),
            "spot": hash_file(job.path("spot.json")),
        },
        outputs=outputs,
        force="score" in force,
        fn=produce,
    )


def run_warp(job: JobStore, force: set[str]) -> str:
    placement = Placement.model_validate_json(job.path("placement.json").read_text())
    creative = job.path(placement.chosen_creative_path)
    surface_path = job.path("track/surface.zip")

    def produce() -> None:
        print("placing: fitting the poster to the surface", file=sys.stderr, flush=True)
        frames, _fps = read_frames(job.path("track/window.mp4"))
        archive = MaskArchive(surface_path)
        try:
            if archive.frame_count != len(frames):
                raise PipelineError("Surface masks and the window video have different frame counts")
            masks = [archive.frame(index) for index in range(len(frames))]
        finally:
            archive.close()
        track = estimate_homographies(frames, masks)
        bad_fraction = 1.0 - float(track.valid.mean())
        if track.jitter_rms > JITTER_BUDGET:
            raise PipelineError(
                f"Placement jitter is {track.jitter_rms:.2f}px, above the {JITTER_BUDGET:.1f}px budget"
            )
        if bad_fraction > MAX_BAD_FRAME_FRACTION:
            raise PipelineError(f"{bad_fraction:.0%} of frames lost the surface track")
        target = job.path("warp/homographies.npz")
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, homographies=track.homographies, corners=track.corners_smooth)
        job.path("warp/metrics.json").write_text(
            json.dumps({"jitter_rms": track.jitter_rms, "bad_frame_fraction": bad_fraction}, indent=2)
        )

    return run_cached(
        job,
        stage="warp",
        version=WARP_VERSION,
        inputs={
            "surface": hash_file(surface_path),
            "creative": hash_file(creative),
            "window": f"{placement.start_s:.3f}-{placement.end_s:.3f}",
        },
        outputs=["warp/homographies.npz", "warp/metrics.json"],
        force="warp" in force,
        fn=produce,
    )


def run_composite(job: JobStore, force: set[str]) -> str:
    placement = Placement.model_validate_json(job.path("placement.json").read_text())
    creative_path = job.path(placement.chosen_creative_path)
    metrics = json.loads(job.path("warp/metrics.json").read_text())

    def produce() -> None:
        print("placing: painting the poster onto the video", file=sys.stderr, flush=True)
        creative = cv2.imread(str(creative_path), cv2.IMREAD_COLOR)
        if creative is None:
            raise PipelineError("The chosen creative could not be read")
        frames, fps = read_frames(job.path("track/window.mp4"))
        surface = MaskArchive(job.path("track/surface.zip"))
        person = MaskArchive(job.path("track/person.zip"))
        try:
            stored = np.load(job.path("warp/homographies.npz"))
            corners = stored["corners"]
            if len(corners) != len(frames):
                raise PipelineError("Homographies do not match the window")
            rendered: list[np.ndarray] = []
            centers = corners.reshape(len(corners), -1, 2).mean(axis=1)
            for index, frame in enumerate(frames):
                previous = centers[index - 1] if index else centers[0]
                motion = (float(centers[index, 0] - previous[0]), float(centers[index, 1] - previous[1]))
                rendered.append(
                    composite_frame(
                        frame,
                        creative,
                        surface.frame(index),
                        person.frame(index),
                        corners[index],
                        motion=motion,
                    )
                )
        finally:
            surface.close()
            person.close()
        write_mp4(job.path("render/output.mp4"), rendered, fps)
        record = RenderRecord(
            placement_id=job.id,
            output_path="render/output.mp4",
            jitter_rms=float(metrics["jitter_rms"]),
        )
        job.write_model("render/render.json", record)

    return run_cached(
        job,
        stage="composite",
        version=COMPOSITE_VERSION,
        inputs={
            "warp": hash_file(job.path("warp/homographies.npz")),
            "creative": hash_file(creative_path),
            "surface": hash_file(job.path("track/surface.zip")),
            "person": hash_file(job.path("track/person.zip")),
        },
        outputs=["render/output.mp4", "render/render.json"],
        force="composite" in force,
        fn=produce,
    )


def run_report(job: JobStore, force: set[str]) -> str:
    def produce() -> None:
        surface = MaskArchive(job.path("track/surface.zip"))
        person = MaskArchive(job.path("track/person.zip"))
        try:
            pairs = ((surface.frame(index), person.frame(index)) for index in range(surface.frame_count))
            visibility = visibility_over_frames(pairs, fps=surface.fps, render_id=job.id)
        finally:
            surface.close()
            person.close()
        job.write_model("visibility.json", visibility)

    return run_cached(
        job,
        stage="report",
        version=REPORT_VERSION,
        inputs={
            "surface": hash_file(job.path("track/surface.zip")),
            "person": hash_file(job.path("track/person.zip")),
            "render": hash_file(job.path("render/render.json")),
        },
        outputs=["visibility.json"],
        force="report" in force,
        fn=produce,
    )


def _thumbnail_frame(source: Path, time_s: float) -> np.ndarray:
    return frame_at(source, time_s)
