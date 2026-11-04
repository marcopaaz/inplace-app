import subprocess
from pathlib import Path

import cv2
import numpy as np

from inplace.constants import JITTER_BUDGET
from inplace.jobs import JobStore
from inplace.mask_archive import save_mask_archive
from inplace.media import write_mp4
from inplace.models import (
    BrandBrief,
    PlaceabilityReport,
    Placement,
    RenderRecord,
    ScenePlan,
    SignalScan,
    SpotMeasurement,
    SpotReport,
    Visibility,
)
from inplace.stages.probe import run_probe
from inplace.stages.run import run_composite, run_report, run_score, run_warp
from inplace.stages.signals import run_signals


def test_local_stages_cache_and_reject_a_short_clip(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:d=1",
            "-f",
            "lavfi",
            "-i",
            "color=c=white:s=320x240:d=1",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
        capture_output=True,
    )
    job = JobStore.create(tmp_path / "jobs", video)
    assert run_probe(job, set()) == "ran"
    assert run_signals(job, set()) == "ran"
    assert run_signals(job, set()) == "cached"
    scan = SignalScan.model_validate_json(job.path("signals.json").read_text())
    assert scan.cut_times
    assert scan.samples

    plan = ScenePlan.model_validate(
        {
            "windows": [
                {"start_s": 0, "end_s": 2, "surface_label": "wall", "sam_prompt": "blank wall"},
            ],
            "moments_to_avoid": [],
        }
    )
    job.write_model("scene_plan.json", plan)
    job.write_model(
        "spot.json",
        SpotReport(
            measurements=[
                SpotMeasurement(
                    start_s=0,
                    end_s=2,
                    sam_prompt="blank wall",
                    occlusion=0.2,
                    aspect_w=4,
                    aspect_h=3,
                    surface_found=True,
                )
            ]
        ),
    )
    assert run_score(job, set()) == "ran"
    report = PlaceabilityReport.model_validate_json(job.path("placeability.json").read_text())
    assert report.status == "stopped"
    assert report.score == 0
    assert report.windows == []
    assert any("3" in reason for reason in report.reasons)


def test_render_stages_place_a_synthetic_quad(tmp_path: Path) -> None:
    frames, surface_masks, person_masks = _quad_clip()
    job = JobStore(tmp_path / "job")
    job.root.mkdir()
    write_mp4(job.path("track/window.mp4"), frames, fps=10)
    save_mask_archive(job.path("track/surface.zip"), surface_masks, fps=10)
    save_mask_archive(job.path("track/person.zip"), person_masks, fps=10)
    creative = np.zeros((80, 100, 3), dtype=np.uint8)
    creative[:, :] = (0, 0, 220)
    job.path("creative").mkdir()
    if not cv2.imwrite(str(job.path("creative/0.png")), creative):
        raise AssertionError("creative was not written")
    job.write_model(
        "placement.json",
        Placement(
            video_id=job.id,
            window_index=0,
            start_s=0,
            end_s=1,
            surface_label="wall",
            sam_prompt="blank wall",
            aspect_w=100,
            aspect_h=80,
            brand=BrandBrief(name="Northline", description="A fictional trail-food brand.", style="flat poster"),
            creative_index=0,
            chosen_creative_path="creative/0.png",
            status="draft",
        ),
    )
    assert run_warp(job, set()) == "ran"
    assert run_warp(job, set()) == "cached"
    assert run_composite(job, set()) == "ran"
    assert run_report(job, set()) == "ran"
    rendered = RenderRecord.model_validate_json(job.path("render/render.json").read_text())
    assert rendered.jitter_rms < JITTER_BUDGET
    assert job.path("render/output.mp4").is_file()
    visibility = Visibility.model_validate_json(job.path("visibility.json").read_text())
    assert visibility.seconds_on_screen > 0
    assert visibility.share_of_time_occluded > 0


def _quad_clip() -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    height, width, count = 180, 240, 8
    texture = np.zeros((80, 100, 3), dtype=np.uint8)
    texture[::8, :] = 255
    texture[:, ::8] = 255
    source = np.array([[0, 0], [99, 0], [99, 79], [0, 79]], dtype=np.float32)
    frames: list[np.ndarray] = []
    surface_masks: list[np.ndarray] = []
    person_masks: list[np.ndarray] = []
    for index in range(count):
        shift_x = 20 + index * 3
        shift_y = 30 + index
        dest = np.array(
            [
                [shift_x, shift_y],
                [shift_x + 90, shift_y + 4],
                [shift_x + 84, shift_y + 70],
                [shift_x + 2, shift_y + 64],
            ],
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(source, dest)
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        warped = cv2.warpPerspective(texture, homography, (width, height))
        mask = cv2.warpPerspective(np.full(texture.shape[:2], 255, np.uint8), homography, (width, height))
        mask = np.where(mask > 127, 255, 0).astype(np.uint8)
        frame[mask > 0] = warped[mask > 0]
        person = np.zeros((height, width), dtype=np.uint8)
        if index < 2:
            person[mask > 0] = 255
        frames.append(frame)
        surface_masks.append(mask)
        person_masks.append(person)
    return frames, surface_masks, person_masks
