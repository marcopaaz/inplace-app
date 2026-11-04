from inplace.stages.score import split_on_cuts, subtract_intervals


def test_avoid_ranges_are_removed_before_scoring() -> None:
    pieces = subtract_intervals(0.0, 20.0, [(5.0, 8.0), (12.0, 13.0)])
    assert pieces == [(0.0, 5.0), (8.0, 12.0), (13.0, 20.0)]


def test_a_cut_splits_the_window() -> None:
    assert split_on_cuts(0.0, 10.0, [4.0, 12.0]) == [(0.0, 4.0), (4.0, 10.0)]
