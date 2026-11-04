import os
import subprocess
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

from inplace.jobs import JobStore
from inplace.models import (
    PlaceabilityReport,
    PlaceabilityStatus,
    PlacementWindow,
    SignalScores,
    Video,
    Visibility,
)
from inplace.pipeline import draft_placement
from inplace.stages.creative import resolve_brand
from inplace.webapp import create_app

JOB_ID = "abc123abc123"


def seed(tmp_path: Path, job_id: str = JOB_ID) -> JobStore:
    root = tmp_path / job_id
    root.mkdir(parents=True)
    (root / "source.mp4").write_bytes(b"source")
    return JobStore(root)


def report_for(
    job_id: str,
    *,
    status: PlaceabilityStatus = "placeable",
    score: float = 82,
) -> PlaceabilityReport:
    windows: list[PlacementWindow] = []
    reasons: list[str] = []
    if status == "stopped":
        reasons = ["under 3 s"]
    else:
        windows.append(
            PlacementWindow(
                start_s=1,
                end_s=9,
                surface_label="table",
                sam_prompt="table top",
                thumbnail="thumbs/0.jpg",
                window_score=score,
                signal_scores=SignalScores(
                    surface_time=0.9,
                    camera_motion=0.8,
                    occlusion=0.85,
                    lighting=0.9,
                    shot_stability=0.88,
                ),
                aspect_w=4,
                aspect_h=3,
            )
        )
    return PlaceabilityReport(
        video_id=job_id,
        score=score if status != "stopped" else 20,
        status=status,
        reasons=reasons,
        windows=windows,
    )


def video_for(job_id: str) -> Video:
    return Video(
        id=job_id,
        file_path="source.mp4",
        duration_s=12,
        fps=24,
        width=1280,
        height=720,
        codec="h264",
    )


def wait_for(client: TestClient, job_id: str, phase: str) -> dict:
    deadline = time.monotonic() + 2
    body: dict | None = None
    while time.monotonic() < deadline:
        body = client.get(f"/api/jobs/{job_id}").json()
        if body["phase"] == phase:
            return body
        time.sleep(0.02)
    raise AssertionError(body)


def test_page_offers_an_upload(tmp_path: Path) -> None:
    response = TestClient(create_app(tmp_path)).get("/")
    assert response.status_code == 200
    assert "Drop an MP4" in response.text
    assert "Make three posters" in response.text
    page = Path(__file__).resolve().parents[1] / "src" / "inplace" / "web" / "index.html"
    assert "Place poster " in page.read_text()


def test_ui_help_lists_the_port() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "-m", "inplace", "ui", "--help"],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--port" in completed.stdout


def test_brands_are_the_samples(tmp_path: Path) -> None:
    payload = TestClient(create_app(tmp_path)).get("/api/brands").json()
    assert {item["name"] for item in payload} == {"Northline", "Lumen Soda", "Paper Orbit"}


def test_rejects_non_mp4_and_empty_uploads(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path))
    notes = client.post("/api/jobs", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert notes.status_code == 400
    assert "MP4" in notes.json()["detail"]
    empty = client.post("/api/jobs", files={"file": ("clip.mp4", b"", "video/mp4")})
    assert empty.status_code == 400
    assert "empty" in empty.json()["detail"]
    assert list(tmp_path.iterdir()) == []


def test_upload_starts_a_check_without_calling_meta(tmp_path: Path, monkeypatch) -> None:
    def fake_check(job: JobStore, client: object, force: set[str]) -> None:
        thumb = job.path("thumbs/0.jpg")
        thumb.parent.mkdir(parents=True, exist_ok=True)
        thumb.write_bytes(b"jpeg")
        job.write_model("video.json", video_for(job.id))
        job.write_model("placeability.json", report_for(job.id))

    monkeypatch.setattr("inplace.webapp.check_job", fake_check)
    monkeypatch.setattr("inplace.webapp.MetaClient", lambda: object())
    client = TestClient(create_app(tmp_path))
    response = client.post("/api/jobs", files={"file": ("clip.mp4", b"not-a-video", "video/mp4")})
    assert response.status_code == 200
    job_id = response.json()["id"]
    body = wait_for(client, job_id, "ready")
    assert body["report"]["score"] == 82
    assert "/file/thumbs/0.jpg?" in body["report"]["windows"][0]["thumbnail"]
    assert body["video"]["width"] == 1280
    assert (tmp_path / job_id / "source.mp4").read_bytes() == b"not-a-video"
    thumb = client.get(body["report"]["windows"][0]["thumbnail"])
    assert thumb.status_code == 200
    assert thumb.content == b"jpeg"


def test_status_follows_files_on_disk(tmp_path: Path) -> None:
    job = seed(tmp_path)
    job.write_model("video.json", video_for(job.id))
    job.path("signals.json").write_text("{}")
    app = create_app(tmp_path)
    app.state.hub.running[job.id] = "check"
    client = TestClient(app)
    checking = client.get(f"/api/jobs/{job.id}").json()
    states = {row["id"]: row["state"] for row in checking["check_stages"]}
    assert checking["phase"] == "checking"
    assert states["probe"] == "done"
    assert states["signals"] == "done"
    assert states["plan"] == "active"
    assert {row["state"] for row in checking["render_stages"]} == {"pending"}

    app.state.hub.running.clear()
    job.write_model("placeability.json", report_for(job.id))
    thumb = job.path("thumbs/0.jpg")
    thumb.parent.mkdir()
    thumb.write_bytes(b"jpeg")
    ready = client.get(f"/api/jobs/{job.id}").json()
    assert ready["phase"] == "ready"
    assert ready["report"]["windows"][0]["surface_label"] == "table"

    for index in range(3):
        path = job.path(f"creative/{index}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"png")
    choose = client.get(f"/api/jobs/{job.id}").json()
    assert choose["phase"] == "choose"
    assert len(choose["creatives"]) == 3

    job.path("track").mkdir()
    job.path("track/window.mp4").write_bytes(b"before")
    job.path("render").mkdir()
    job.path("render/output.mp4").write_bytes(b"after")
    job.write_model(
        "visibility.json",
        Visibility(render_id=job.id, seconds_on_screen=4.2, average_share_of_frame=0.12, share_of_time_occluded=0.18),
    )
    done = client.get(f"/api/jobs/{job.id}").json()
    assert done["phase"] == "done"
    assert "/file/track/window.mp4?" in done["before_url"]
    assert "/file/render/output.mp4?" in done["after_url"]
    assert done["visibility"]["seconds_on_screen"] == 4.2


def test_stage_error_is_reported(tmp_path: Path) -> None:
    job = seed(tmp_path)
    job.write_error("probe", "Codec gif is not supported")
    body = TestClient(create_app(tmp_path)).get(f"/api/jobs/{job.id}").json()
    assert body["phase"] == "error"
    assert body["error"] == "Codec gif is not supported"
    assert body["check_stages"][0]["state"] == "error"
    assert body["check_stages"][1]["state"] == "pending"


def test_poster_request_is_guarded(tmp_path: Path) -> None:
    job = seed(tmp_path)
    app = create_app(tmp_path)
    client = TestClient(app)
    missing = client.post(f"/api/jobs/{job.id}/posters", json={"window": 0, "brand": "Northline"})
    assert missing.status_code == 400
    assert "placeability" in missing.json()["detail"]

    job.write_model("placeability.json", report_for(job.id, status="stopped"))
    stopped = client.post(f"/api/jobs/{job.id}/posters", json={"window": 0, "brand": "Northline"})
    assert stopped.status_code == 400
    assert "under 3 s" in stopped.json()["detail"]
    assert not job.path("placement.json").exists()

    job.write_model("placeability.json", report_for(job.id))
    unnamed = client.post(f"/api/jobs/{job.id}/posters", json={"window": 0, "brand": "Acme"})
    assert unnamed.status_code == 400
    assert job.id not in app.state.hub.running

    out_of_range = client.post(f"/api/jobs/{job.id}/posters", json={"window": 4, "brand": "Northline"})
    assert out_of_range.status_code == 400
    assert "Window 4" in out_of_range.json()["detail"]

    app.state.hub.running[job.id] = "check"
    busy = client.post(f"/api/jobs/{job.id}/posters", json={"window": 0, "brand": "Northline"})
    assert busy.status_code == 409


def test_poster_and_place_run_in_the_background(tmp_path: Path, monkeypatch) -> None:
    job = seed(tmp_path)
    job.write_model("placeability.json", report_for(job.id))

    def fake_creative(target: JobStore, client: object, force: set[str]) -> None:
        for index in range(3):
            path = target.path(f"creative/{index}.png")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"poster-{index}".encode())

    def fake_render(
        target: JobStore,
        *,
        window_index: int,
        brand: object,
        creative_index: int,
        client: object,
        force: set[str],
    ) -> Visibility:
        called["creative"] = creative_index
        called["window"] = window_index
        called["brand"] = brand.name
        target.path("track").mkdir(parents=True, exist_ok=True)
        target.path("track/window.mp4").write_bytes(b"before")
        target.path("render").mkdir(parents=True, exist_ok=True)
        target.path("render/output.mp4").write_bytes(b"after")
        visibility = Visibility(
            render_id=target.id,
            seconds_on_screen=4.2,
            average_share_of_frame=0.12,
            share_of_time_occluded=0.18,
        )
        target.write_model("visibility.json", visibility)
        return visibility

    called: dict[str, object] = {}
    monkeypatch.setattr("inplace.webapp.run_creative", fake_creative)
    monkeypatch.setattr("inplace.webapp.render", fake_render)
    monkeypatch.setattr("inplace.webapp.MetaClient", lambda: object())
    client = TestClient(create_app(tmp_path))
    posters = client.post(
        f"/api/jobs/{job.id}/posters",
        json={"window": 0, "brand": "Northline", "description": "", "style": ""},
    )
    assert posters.status_code == 200
    choose = wait_for(client, job.id, "choose")
    assert len(choose["creatives"]) == 3
    assert choose["placement"]["brand"]["name"] == "Northline"

    placed = client.post(f"/api/jobs/{job.id}/render", json={"creative": 2})
    assert placed.status_code == 200
    done = wait_for(client, job.id, "done")
    assert called == {"creative": 2, "window": 0, "brand": "Northline"}
    assert "track/window.mp4?" in done["before_url"]
    assert client.get(done["after_url"]).content == b"after"


def test_render_requires_posters(tmp_path: Path) -> None:
    job = seed(tmp_path)
    job.write_model("placeability.json", report_for(job.id))
    draft_placement(job, window_index=0, brand=resolve_brand("Northline", "", ""), creative_index=0)
    app = create_app(tmp_path)
    response = TestClient(app).post(f"/api/jobs/{job.id}/render", json={"creative": 1})
    assert response.status_code == 400
    assert "posters" in response.json()["detail"]
    assert job.id not in app.state.hub.running


def test_files_stay_inside_the_job(tmp_path: Path) -> None:
    job = seed(tmp_path)
    (job.root / "thumbs").mkdir()
    (job.root / "thumbs" / "0.jpg").write_bytes(b"jpeg")
    paid = job.root / "paid" / "sam"
    paid.mkdir(parents=True)
    (paid / "secret.txt").write_bytes(b"secret")
    outside = tmp_path / "secret.txt"
    outside.write_text("nope")
    client = TestClient(create_app(tmp_path))
    assert client.get(f"/api/jobs/{job.id}/file/thumbs/0.jpg").content == b"jpeg"
    assert client.get(f"/api/jobs/{job.id}/file/paid/sam/secret.txt").status_code == 404
    assert client.get(f"/api/jobs/{job.id}/file/creative/3.png").status_code == 404
    escaped = client.get(f"/api/jobs/{job.id}/file/thumbs/../../secret.txt")
    assert escaped.status_code == 404
    assert b"nope" not in escaped.content
    encoded = client.get(f"/api/jobs/{job.id}/file/thumbs/%2e%2e/%2e%2e/secret.txt")
    assert encoded.status_code == 404
    assert b"nope" not in encoded.content
    assert client.get("/api/jobs/not-a-real-id").status_code == 404
    assert client.get("/api/jobs/abc123abc124").status_code == 404
