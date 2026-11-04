from inplace.models import MomentToAvoid, ScenePlan, SceneWindow, SignalSample, SignalScan, SpotMeasurement, SpotReport
from inplace.stages.plan import parse_scene_plan
from inplace.stages.score import assemble_windows, placement_spans


def test_spark_json_parses_and_cuts_split_windows() -> None:
    plan = parse_scene_plan(
        '{"windows":[{"start_s":0,"end_s":20,"surface_label":"wall","sam_prompt":"blank wall"}],'
        '"moments_to_avoid":[{"start_s":6,"end_s":9,"reason":"sensitive"}]}'
    )
    scan = SignalScan(
        fps=10,
        samples=[SignalSample(time_s=float(second), flow=0.0, luminance=10.0) for second in range(20)],
        cut_times=[12.0],
    )
    spots = SpotReport(
        measurements=[
            SpotMeasurement(
                start_s=start,
                end_s=end,
                sam_prompt="blank wall",
                occlusion=0.2,
                aspect_w=4,
                aspect_h=3,
                surface_found=True,
            )
            for start, end in ((0.0, 6.0), (9.0, 12.0), (12.0, 20.0))
        ]
    )
    windows = assemble_windows(plan, scan, spots, 20.0)
    assert [(window.start_s, window.end_s) for window in windows] == [(0.0, 6.0), (9.0, 12.0), (12.0, 20.0)]
    assert all(window.surface_found for window in windows)
    assert isinstance(plan.windows[0], SceneWindow)
    assert isinstance(plan.moments_to_avoid[0], MomentToAvoid)


def test_a_long_window_is_cut_to_the_calmest_15_seconds() -> None:
    plan = ScenePlan(
        windows=[
            SceneWindow(start_s=0, end_s=170, surface_label="door", sam_prompt="white door"),
            SceneWindow(start_s=0, end_s=170, surface_label="wall", sam_prompt="blank wall"),
        ],
        moments_to_avoid=[],
    )
    samples = []
    for second in range(170):
        calm = 80 <= second < 95
        samples.append(SignalSample(time_s=float(second), flow=0.2 if calm else 5.0, luminance=10.0))
    scan = SignalScan(fps=4, samples=samples, cut_times=[])
    spans = placement_spans(plan, scan, 177.9)
    assert len(spans) == 2
    for start_s, end_s, _label, _prompt in spans:
        assert end_s - start_s == 15
        assert 80 <= start_s < 95
        assert end_s <= 110
