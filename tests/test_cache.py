from pathlib import Path

import pytest

from inplace.jobs import JobStore, cache_key
from inplace.runner import invalidate, run_cached


def test_cache_key_ignores_input_order() -> None:
    left = cache_key(stage="warp", version="1", inputs={"video": "abc", "masks": "def"})
    right = cache_key(stage="warp", version="1", inputs={"masks": "def", "video": "abc"})
    assert left == right


def test_cache_key_changes_when_version_or_model_changes() -> None:
    base = cache_key(stage="track", version="1", inputs={"model": "sam-3.1", "prompt": "wall"})
    assert base != cache_key(stage="track", version="2", inputs={"model": "sam-3.1", "prompt": "wall"})
    assert base != cache_key(stage="track", version="1", inputs={"model": "sam-3.0", "prompt": "wall"})


def test_runner_skips_when_outputs_and_key_match(tmp_path: Path) -> None:
    job = JobStore(tmp_path / "job")
    job.root.mkdir()
    calls = {"n": 0}

    def produce() -> None:
        calls["n"] += 1
        job.path("warp/homographies.npz").parent.mkdir(parents=True, exist_ok=True)
        job.path("warp/homographies.npz").write_bytes(b"h")

    first = run_cached(
        job,
        stage="warp",
        version="1",
        inputs={"masks": "aaa"},
        outputs=["warp/homographies.npz"],
        force=False,
        fn=produce,
    )
    second = run_cached(
        job,
        stage="warp",
        version="1",
        inputs={"masks": "aaa"},
        outputs=["warp/homographies.npz"],
        force=False,
        fn=produce,
    )
    assert first == "ran"
    assert second == "cached"
    assert calls["n"] == 1


def test_failed_stage_is_not_a_cache_hit_and_keeps_paid_stream(tmp_path: Path) -> None:
    job = JobStore(tmp_path / "job")
    job.root.mkdir()

    def blow_up() -> None:
        paid = job.path("paid/sam/wall.txt")
        paid.parent.mkdir(parents=True, exist_ok=True)
        paid.write_text("stream")
        derived = job.path("track/surface.npz")
        derived.parent.mkdir(parents=True, exist_ok=True)
        derived.write_bytes(b"masks")
        raise RuntimeError("parse failed")

    with pytest.raises(RuntimeError, match="parse failed"):
        run_cached(
            job,
            stage="track",
            version="1",
            inputs={"prompt": "wall"},
            outputs=["track/surface.npz"],
            force=False,
            fn=blow_up,
        )

    assert not job.is_cached("track", cache_key(stage="track", version="1", inputs={"prompt": "wall"}), ["track/surface.npz"])
    assert not job.path("track/surface.npz").exists()
    assert job.path("paid/sam/wall.txt").read_text() == "stream"
    assert (job.stage_dir("track") / "error.json").is_file()


def test_force_track_drops_warp_and_paid_sam_but_not_creative(tmp_path: Path) -> None:
    job = JobStore(tmp_path / "job")
    job.root.mkdir()
    job.path("warp/homographies.npz").parent.mkdir(parents=True)
    job.path("warp/homographies.npz").write_bytes(b"h")
    job.path("creative/0.png").parent.mkdir(parents=True)
    job.path("creative/0.png").write_bytes(b"png")
    job.path("paid/sam/wall.txt").parent.mkdir(parents=True)
    job.path("paid/sam/wall.txt").write_text("stream")
    job.path("paid/creative/0.png").parent.mkdir(parents=True)
    job.path("paid/creative/0.png").write_bytes(b"png")
    job.write_manifest(stage="warp", cache_key="k", version="1", outputs=["warp/homographies.npz"])
    job.write_manifest(stage="creative", cache_key="k", version="1", outputs=["creative/0.png"])
    job.write_manifest(stage="track", cache_key="k", version="1", outputs=["track/surface.npz"])

    invalidate(job, "track")

    assert not job.path("warp/homographies.npz").exists()
    assert not job.path("paid/sam/wall.txt").exists()
    assert job.path("creative/0.png").read_bytes() == b"png"
    assert job.path("paid/creative/0.png").read_bytes() == b"png"
