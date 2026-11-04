from __future__ import annotations

from pathlib import Path

from inplace.errors import PipelineError
from inplace.jobs import JobStore
from inplace.meta_client import MetaClient
from inplace.models import BrandBrief, PlaceabilityReport, Placement, Visibility
from inplace.stages.creative import run_creative
from inplace.stages.plan import run_plan
from inplace.stages.probe import run_probe
from inplace.stages.run import run_composite, run_report, run_score, run_warp
from inplace.stages.signals import run_signals
from inplace.stages.spot import run_spot
from inplace.stages.track import render_bounds, run_track


def check_job(job: JobStore, client: MetaClient, force: set[str]) -> PlaceabilityReport:
    run_probe(job, force)
    run_signals(job, force)
    run_plan(job, client, force)
    run_spot(job, client, force)
    run_score(job, force)
    return PlaceabilityReport.model_validate_json(job.path("placeability.json").read_text())


def check(source: Path, jobs_root: Path, client: MetaClient, force: set[str]) -> tuple[JobStore, PlaceabilityReport]:
    job = JobStore.create(jobs_root, source)
    return job, check_job(job, client, force)


def draft_placement(
    job: JobStore,
    *,
    window_index: int,
    brand: BrandBrief,
    creative_index: int,
) -> Placement:
    if not job.path("placeability.json").is_file():
        raise PipelineError("This job has no placeability report. Run check first.")
    report = PlaceabilityReport.model_validate_json(job.path("placeability.json").read_text())
    if report.status == "stopped":
        reason = report.reasons[0] if report.reasons else "This video is not placeable"
        raise PipelineError(reason)
    if window_index < 0 or window_index >= len(report.windows):
        raise PipelineError(f"Window {window_index} is not in the placeability report")
    if creative_index not in {0, 1, 2}:
        raise PipelineError("Pick creative 0, 1, or 2")
    window = report.windows[window_index]
    if window.aspect_w <= 0 or window.aspect_h <= 0:
        raise PipelineError("This window has no surface aspect. Run the placeability check again.")
    start_s, end_s = render_bounds(window.start_s, window.end_s)
    placement = Placement(
        video_id=job.id,
        window_index=window_index,
        start_s=start_s,
        end_s=end_s,
        surface_label=window.surface_label,
        sam_prompt=window.sam_prompt,
        aspect_w=window.aspect_w,
        aspect_h=window.aspect_h,
        brand=brand,
        creative_index=creative_index,
        chosen_creative_path=f"creative/{creative_index}.png",
        status="draft",
    )
    job.write_model("placement.json", placement)
    return placement


def render(
    job: JobStore,
    *,
    window_index: int,
    brand: BrandBrief,
    creative_index: int,
    client: MetaClient,
    force: set[str],
) -> Visibility:
    placement = draft_placement(
        job,
        window_index=window_index,
        brand=brand,
        creative_index=creative_index,
    )
    run_track(job, client, force)
    run_creative(job, client, force)
    run_warp(job, force)
    run_composite(job, force)
    run_report(job, force)
    placement.status = "rendered"
    job.write_model("placement.json", placement)
    return Visibility.model_validate_json(job.path("visibility.json").read_text())


def format_report(job: JobStore, report: PlaceabilityReport) -> str:
    lines = [f"job: {job.root}", f"score: {report.score:.0f}  {report.status}"]
    if report.reasons:
        lines.append("reasons:")
        lines.extend(f"- {reason}" for reason in report.reasons)
    if report.windows:
        lines.append("windows:")
        for index, window in enumerate(report.windows):
            lines.append(
                f"{index}  {window.start_s:.2f}-{window.end_s:.2f}s  {window.surface_label}  "
                f"prompt={window.sam_prompt!r}  score={window.window_score:.0f}"
            )
    else:
        lines.append("windows: none")
    return "\n".join(lines)


def format_render(job: JobStore, visibility: Visibility) -> str:
    return "\n".join(
        [
            f"job: {job.root}",
            f"video: {job.path('render/output.mp4')}",
            f"seconds on screen: {visibility.seconds_on_screen:.2f}",
            f"average share of frame: {visibility.average_share_of_frame:.3f}",
            f"share of time occluded: {visibility.share_of_time_occluded:.3f}",
        ]
    )
