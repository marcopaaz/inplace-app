from inplace.constants import FLOW_LIMIT, OCCLUSION_DROP
from inplace.stages.score import WindowInput, score_window, summarize_windows


def _window(**overrides: float | str | bool) -> WindowInput:
    values: dict[str, float | str | bool] = {
        "start_s": 0.0,
        "end_s": 10.0,
        "surface_label": "blank wall",
        "sam_prompt": "blank wall",
        "flow": 0.0,
        "luminance_variance": 0.0,
        "occlusion": 0.2,
        "has_cut": False,
        "aspect_w": 4,
        "aspect_h": 3,
        "surface_found": True,
    }
    values.update(overrides)
    return WindowInput(**values)  # type: ignore[arg-type]


def test_perfect_window_scores_100() -> None:
    scored = score_window(_window())
    assert scored.kept
    assert scored.score == 100


def test_three_second_window_stays_and_shorter_window_is_dropped() -> None:
    kept = score_window(_window(end_s=3.0))
    assert kept.kept
    assert kept.score == 79
    dropped = score_window(_window(end_s=2.9))
    assert not dropped.kept
    assert dropped.drop_reason is not None
    assert "3" in dropped.drop_reason


def test_shake_and_occlusion_hard_gates() -> None:
    at_limit = score_window(_window(flow=FLOW_LIMIT))
    assert at_limit.kept
    assert at_limit.signals.camera_motion == 0
    over = score_window(_window(flow=FLOW_LIMIT + 0.01))
    assert not over.kept
    assert over.drop_reason is not None
    assert "camera" in over.drop_reason.lower() or "shake" in over.drop_reason.lower()

    at_occlusion = score_window(_window(occlusion=OCCLUSION_DROP))
    assert at_occlusion.kept
    covered = score_window(_window(occlusion=OCCLUSION_DROP + 0.01))
    assert not covered.kept


def test_missing_surface_is_dropped() -> None:
    dropped = score_window(_window(surface_found=False))
    assert not dropped.kept
    assert dropped.drop_reason is not None
    assert "surface" in dropped.drop_reason.lower()


def test_video_score_is_the_best_kept_window_and_thresholds() -> None:
    report = summarize_windows(
        "vid",
        [
            score_window(_window(end_s=2.0, surface_label="too short")),
            score_window(_window(start_s=4.0, end_s=7.0, surface_label="ok")),
        ],
    )
    assert report.score == 79
    assert report.status == "placeable"
    assert len(report.windows) == 1
    assert report.windows[0].surface_label == "ok"

    warning = summarize_windows("vid", [score_window(_window(flow=FLOW_LIMIT, luminance_variance=40))])
    assert warning.status == "warning"
    assert 40 <= warning.score < 70

    stopped = summarize_windows("vid", [score_window(_window(end_s=2.0))])
    assert stopped.status == "stopped"
    assert stopped.score == 0
    assert stopped.windows == []
    assert stopped.reasons


def test_reasons_are_the_two_lowest_signals() -> None:
    scored = score_window(_window(end_s=3.0, flow=FLOW_LIMIT))
    report = summarize_windows("vid", [scored])
    assert len(report.reasons) == 2
    assert "camera" in report.reasons[0].lower() or "surface" in report.reasons[0].lower()
