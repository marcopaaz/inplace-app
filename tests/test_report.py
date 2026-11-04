import numpy as np

from inplace.stages.report import visibility_from_masks


def test_visibility_counts_screen_time_share_and_occlusion() -> None:
    surface = np.zeros((10, 4, 4), dtype=np.uint8)
    occluder = np.zeros_like(surface)
    surface[:, :2, :] = 255
    occluder[0, :2, :] = 255
    occluder[1, :2, :] = 255
    metrics = visibility_from_masks(surface, occluder, fps=10)
    assert metrics.seconds_on_screen == 0.8
    assert metrics.average_share_of_frame == 0.4
    assert metrics.share_of_time_occluded == 0.2
