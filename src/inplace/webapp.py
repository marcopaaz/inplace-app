from __future__ import annotations

import json
import re
import sys
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Never

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ValidationError

from inplace.errors import PipelineError, VideoRejected
from inplace.jobs import JobStore
from inplace.meta_client import MetaClient
from inplace.models import PlaceabilityReport, Placement, RenderRecord, Video, Visibility
from inplace.pipeline import check_job, draft_placement, render
from inplace.stages.creative import SAMPLE_BRANDS, resolve_brand, run_creative

PAGE = Path(__file__).resolve().parent / "web" / "index.html"
Activity = Literal["check", "posters", "render"]
Phase = Literal["checking", "ready", "making_posters", "choose", "rendering", "done", "error"]
StageState = Literal["pending", "active", "done", "error"]

CHECK_STAGES = (
    ("probe", "Reading the file", "video.json"),
    ("signals", "Measuring camera motion", "signals.json"),
    ("plan", "Watching the scene", "scene_plan.json"),
    ("spot", "Checking the surfaces", "spot.json"),
    ("score", "Scoring placeability", "placeability.json"),
)
RENDER_STAGES = (
    ("track", "Tracking the surface", "track/surface.zip"),
    ("creative", "Generating posters", "creative/2.png"),
    ("warp", "Fitting the poster", "warp/homographies.npz"),
    ("composite", "Compositing", "render/output.mp4"),
    ("report", "Measuring visibility", "visibility.json"),
)
_JOB_ID = re.compile(r"^[0-9a-f]{12}$")
_ALLOWED_FILE = re.compile(
    r"^(?:source\.mp4|track/window\.mp4|render/output\.mp4|thumbs/[0-9]+\.jpg|creative/[012]\.png)$"
)


class Hub:
    def __init__(self, jobs_root: Path) -> None:
        self.jobs_root = Path(jobs_root)
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        self.running: dict[str, Activity] = {}
        self.failures: dict[str, str] = {}
        self.lock = threading.Lock()


class PosterRequest(BaseModel):
    window: int
    brand: str
    description: str = ""
    style: str = ""


class RenderRequest(BaseModel):
    creative: int


def create_app(jobs_root: Path) -> FastAPI:
    hub = Hub(jobs_root)
    app = FastAPI(title="Inplace")
    app.state.hub = hub

    @app.get("/")
    def index() -> FileResponse:
        if not PAGE.is_file():
            raise HTTPException(status_code=500, detail="The page file is missing")
        return FileResponse(PAGE, media_type="text/html; charset=utf-8", headers={"Cache-Control": "no-store"})

    @app.get("/api/brands")
    def brands() -> list[dict[str, str]]:
        return [brief.model_dump() for brief in SAMPLE_BRANDS.values()]

    @app.post("/api/jobs")
    async def create_job(file: UploadFile) -> dict[str, str]:
        name = file.filename or ""
        if Path(name).suffix.lower() != ".mp4":
            raise HTTPException(status_code=400, detail="Only MP4 uploads are accepted")
        handle = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        try:
            while True:
                chunk = await file.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
            if handle.tell() == 0:
                raise HTTPException(status_code=400, detail="The file is empty")
            handle.close()
            try:
                job = JobStore.create(hub.jobs_root, Path(handle.name))
            except VideoRejected as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            handle.close()
            Path(handle.name).unlink(missing_ok=True)
        _start(hub, job, "check", lambda: _run_check(job))
        return {"id": job.id}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict[str, object]:
        job = _open(hub, job_id)
        with hub.lock:
            activity = hub.running.get(job_id)
            failure = hub.failures.get(job_id)
        return describe_job(job, activity=activity, failure=failure)

    @app.get("/api/jobs/{job_id}/file/{relative:path}")
    def job_file(job_id: str, relative: str) -> FileResponse:
        job = _open(hub, job_id)
        return FileResponse(_safe_file(job, relative))

    @app.post("/api/jobs/{job_id}/posters")
    def start_posters(job_id: str, body: PosterRequest) -> dict[str, str]:
        job = _open(hub, job_id)
        _require_idle(hub, job_id)
        try:
            brand = resolve_brand(body.brand, body.description, body.style)
            draft_placement(job, window_index=body.window, brand=brand, creative_index=0)
        except (PipelineError, ValidationError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _start(hub, job, "posters", lambda: _run_posters(job))
        return {"id": job.id}

    @app.post("/api/jobs/{job_id}/render")
    def start_render(job_id: str, body: RenderRequest) -> dict[str, str]:
        job = _open(hub, job_id)
        _require_idle(hub, job_id)
        posters_ready = job.path("placement.json").is_file() and all(
            job.path(f"creative/{index}.png").is_file() for index in range(3)
        )
        if not posters_ready:
            raise HTTPException(status_code=400, detail="Make the posters before placing one")
        try:
            placement = Placement.model_validate_json(job.path("placement.json").read_text())
            draft_placement(
                job,
                window_index=placement.window_index,
                brand=placement.brand,
                creative_index=body.creative,
            )
        except (PipelineError, ValidationError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _start(hub, job, "render", lambda: _run_render(job, body.creative))
        return {"id": job.id}

    return app


def describe_job(job: JobStore, *, activity: Activity | None, failure: str | None) -> dict[str, object]:
    phase = _phase(job, activity=activity, failure=failure)
    error = None if activity is not None or phase == "done" else _stage_failure(job) or failure
    if phase == "error" and error is None:
        error = "The check stopped before it finished. Upload the video again."
    report = _report(job)
    return {
        "id": job.id,
        "phase": phase,
        "error": error,
        "check_stages": _rows(job, CHECK_STAGES, activity=activity, group="check"),
        "render_stages": _rows(job, RENDER_STAGES, activity=activity, group="render"),
        "report": report,
        "placement": _placement(job),
        "video": _video(job),
        "creatives": [
            _file_url(job, f"creative/{index}.png")
            for index in range(3)
            if job.path(f"creative/{index}.png").is_file()
        ],
        "source_url": _optional_url(job, "source.mp4"),
        "before_url": _optional_url(job, "track/window.mp4"),
        "after_url": _optional_url(job, "render/output.mp4"),
        "visibility": _visibility(job),
        "render": _render_record(job),
    }


def serve(jobs_root: Path, host: str = "127.0.0.1", port: int = 8000) -> None:
    print(f"Inplace is at http://{host}:{port}", flush=True)
    uvicorn.run(create_app(jobs_root), host=host, port=port)


def _run_check(job: JobStore) -> None:
    check_job(job, MetaClient(), set())


def _run_posters(job: JobStore) -> None:
    run_creative(job, MetaClient(), set())


def _run_render(job: JobStore, creative_index: int) -> None:
    placement = Placement.model_validate_json(job.path("placement.json").read_text())
    render(
        job,
        window_index=placement.window_index,
        brand=placement.brand,
        creative_index=creative_index,
        client=MetaClient(),
        force=set(),
    )


def _start(hub: Hub, job: JobStore, activity: Activity, fn: Callable[[], None]) -> None:
    with hub.lock:
        if job.id in hub.running:
            raise HTTPException(status_code=409, detail="This video is already being processed")
        hub.running[job.id] = activity
        hub.failures.pop(job.id, None)

    def guard() -> None:
        try:
            fn()
        except Exception as exc:
            with hub.lock:
                hub.failures[job.id] = str(exc)
            print(f"job {job.id} failed: {exc}", file=sys.stderr)
        finally:
            with hub.lock:
                hub.running.pop(job.id, None)

    threading.Thread(target=guard, name=f"inplace-{job.id}-{activity}", daemon=True).start()


def _require_idle(hub: Hub, job_id: str) -> None:
    with hub.lock:
        if job_id in hub.running:
            raise HTTPException(status_code=409, detail="This video is already being processed")


def _open(hub: Hub, job_id: str) -> JobStore:
    if _JOB_ID.fullmatch(job_id) is None:
        raise HTTPException(status_code=404, detail="No job with that id")
    try:
        return JobStore.open(hub.jobs_root, job_id)
    except PipelineError as exc:
        raise HTTPException(status_code=404, detail="No job with that id") from exc


def _safe_file(job: JobStore, relative: str) -> Path:
    if relative.startswith("/") or ".." in Path(relative).parts or _ALLOWED_FILE.fullmatch(relative) is None:
        raise HTTPException(status_code=404, detail="File not found")
    root = job.root.resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return target


def _phase(job: JobStore, *, activity: Activity | None, failure: str | None) -> Phase:
    if activity == "check":
        return "checking"
    if activity == "posters":
        return "making_posters"
    if activity == "render":
        return "rendering"
    if activity is not None:
        unexpected: Never = activity
        raise AssertionError(unexpected)
    if job.path("visibility.json").is_file():
        return "done"
    if _stage_failure(job) or failure:
        return "error"
    report = job.path("placeability.json")
    if report.is_file() and all(job.path(f"creative/{index}.png").is_file() for index in range(3)):
        return "choose"
    if report.is_file():
        return "ready"
    return "error"


def _rows(
    job: JobStore,
    stages: tuple[tuple[str, str, str], ...],
    *,
    activity: Activity | None,
    group: Literal["check", "render"],
) -> list[dict[str, str]]:
    if group == "check":
        activate = activity == "check"
        only: str | None = None
    elif group == "render":
        activate = activity == "posters" or activity == "render"
        only = "creative" if activity == "posters" else None
    else:
        unexpected: Never = group
        raise AssertionError(unexpected)
    rows: list[dict[str, str]] = []
    claimed = False
    blocked = False
    for name, label, artifact in stages:
        if job.path(artifact).is_file():
            state: StageState = "done"
        elif (job.stage_dir(name) / "error.json").is_file():
            state = "error"
            blocked = True
        elif activate and not claimed and not blocked and (only is None or name == only):
            state = "active"
            claimed = True
        else:
            state = "pending"
        rows.append({"id": name, "label": label, "state": state})
    return rows


def _stage_failure(job: JobStore) -> str | None:
    for name, _label, artifact in (*CHECK_STAGES, *RENDER_STAGES):
        if job.path(artifact).is_file():
            continue
        error_path = job.stage_dir(name) / "error.json"
        if not error_path.is_file():
            continue
        try:
            payload = json.loads(error_path.read_text())
        except json.JSONDecodeError:
            return "A stage failed"
        message = payload.get("message") if isinstance(payload, dict) else None
        return message if isinstance(message, str) and message else "A stage failed"
    return None


def _report(job: JobStore) -> dict[str, object] | None:
    path = job.path("placeability.json")
    if not path.is_file():
        return None
    data = PlaceabilityReport.model_validate_json(path.read_text()).model_dump()
    windows = data.get("windows")
    if isinstance(windows, list):
        for window in windows:
            if not isinstance(window, dict):
                continue
            thumbnail = window.get("thumbnail")
            if isinstance(thumbnail, str) and _ALLOWED_FILE.fullmatch(thumbnail) and job.path(thumbnail).is_file():
                window["thumbnail"] = _file_url(job, thumbnail)
            else:
                window["thumbnail"] = None
    return data


def _placement(job: JobStore) -> dict[str, object] | None:
    path = job.path("placement.json")
    if not path.is_file():
        return None
    return Placement.model_validate_json(path.read_text()).model_dump()


def _video(job: JobStore) -> dict[str, float | int] | None:
    path = job.path("video.json")
    if not path.is_file():
        return None
    video = Video.model_validate_json(path.read_text())
    return {"duration_s": video.duration_s, "width": video.width, "height": video.height, "fps": video.fps}


def _visibility(job: JobStore) -> dict[str, object] | None:
    path = job.path("visibility.json")
    if not path.is_file():
        return None
    return Visibility.model_validate_json(path.read_text()).model_dump()


def _render_record(job: JobStore) -> dict[str, object] | None:
    path = job.path("render/render.json")
    if not path.is_file():
        return None
    return RenderRecord.model_validate_json(path.read_text()).model_dump()


def _optional_url(job: JobStore, relative: str) -> str | None:
    if not job.path(relative).is_file():
        return None
    return _file_url(job, relative)


def _file_url(job: JobStore, relative: str) -> str:
    path = job.path(relative)
    version = int(path.stat().st_mtime) if path.is_file() else 0
    return f"/api/jobs/{job.id}/file/{relative}?v={version}"
