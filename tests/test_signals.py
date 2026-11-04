import numpy as np

from inplace.stages.creative import muse_size
from inplace.stages.signals import median_flow


def test_muse_size_picks_a_supported_canvas() -> None:
    assert muse_size(4, 3) == "1536x1024"
    assert muse_size(3, 4) == "1024x1536"
    assert muse_size(1, 1) == "1024x1024"
    assert muse_size(256, 526) == "1024x1536"


def test_textured_shift_has_flow_and_a_static_pair_does_not() -> None:
    generator = np.random.default_rng(0)
    base = generator.integers(0, 255, size=(64, 80), dtype=np.uint8)
    assert median_flow([base, base]) < 0.2
    assert median_flow([base, np.roll(base, 6, axis=1)]) > 2
