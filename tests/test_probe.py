import subprocess
from pathlib import Path

from inplace.stages.probe import rejection_reason


def test_rejection_names_the_failed_rule() -> None:
    assert rejection_reason(suffix=".mov", duration_s=1, width=100, height=100, codec="h264")
    assert "MP4" in (rejection_reason(suffix=".mov", duration_s=1, width=100, height=100, codec="h264") or "")
    long = rejection_reason(suffix=".mp4", duration_s=181, width=1920, height=1080, codec="h264")
    assert long is not None and "3 minutes" in long
    big = rejection_reason(suffix=".mp4", duration_s=10, width=3840, height=2160, codec="h264")
    assert big is not None and "1080p" in big
    codec = rejection_reason(suffix=".mp4", duration_s=10, width=1280, height=720, codec="vp9")
    assert codec is not None and "vp9" in codec
    portrait = rejection_reason(suffix=".mp4", duration_s=10, width=1080, height=1920, codec="h264")
    assert portrait is None
    ok = rejection_reason(suffix=".mp4", duration_s=180, width=1920, height=1080, codec="hevc")
    assert ok is None


def test_probe_reads_a_real_h264_mp4(tmp_path: Path) -> None:
    from inplace.stages.probe import probe_video

    path = tmp_path / "clip.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=320x240:d=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    video = probe_video(path)
    assert video.width == 320
    assert video.height == 240
    assert video.codec == "h264"
    assert 0.9 <= video.duration_s <= 1.2
    assert video.fps > 0
