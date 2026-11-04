from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from inplace.models import Visibility


def visibility_from_masks(
    surface: np.ndarray,
    occluder: np.ndarray,
    fps: float,
    render_id: str = "",
) -> Visibility:
    pairs = ((surface[index], occluder[index]) for index in range(surface.shape[0]))
    return visibility_over_frames(pairs, fps=fps, render_id=render_id)


def visibility_over_frames(
    frames: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    fps: float,
    render_id: str = "",
) -> Visibility:
    shares: list[float] = []
    occlusions: list[float] = []
    on_screen = 0
    for surface, occluder in frames:
        frame_pixels = surface.shape[0] * surface.shape[1]
        surface_on = surface > 0
        occluder_on = occluder > 0
        visible = surface_on & ~occluder_on
        shares.append(float(visible.sum()) / frame_pixels if frame_pixels else 0.0)
        if visible.any():
            on_screen += 1
        if surface_on.any():
            occlusions.append(float((surface_on & occluder_on).sum()) / float(surface_on.sum()))
    safe_fps = fps if fps > 0 else 1.0
    return Visibility(
        render_id=render_id,
        seconds_on_screen=on_screen / safe_fps,
        average_share_of_frame=float(np.mean(shares)) if shares else 0.0,
        share_of_time_occluded=float(np.mean(occlusions)) if occlusions else 0.0,
    )
