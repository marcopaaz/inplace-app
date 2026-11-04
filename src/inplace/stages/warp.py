from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from inplace.errors import PipelineError


@dataclass
class HomographyTrack:
    corners_raw: np.ndarray
    corners_smooth: np.ndarray
    homographies: np.ndarray
    jitter_rms: float
    valid: np.ndarray


def smooth_tracks(corners: np.ndarray, window: int = 5) -> np.ndarray:
    values = np.asarray(corners, dtype=np.float64)
    count = values.shape[0]
    if count == 0 or window <= 1:
        return values.copy()
    kernel = window if window % 2 == 1 else window + 1
    pad = kernel // 2
    if count == 1:
        return values.copy()
    head_slope = values[1] - values[0]
    tail_slope = values[-1] - values[-2]
    head = values[0] - head_slope * np.arange(pad, 0, -1)[:, None, None]
    tail = values[-1] + tail_slope * np.arange(1, pad + 1)[:, None, None]
    padded = np.concatenate([head, values, tail], axis=0)
    smoothed = np.empty_like(values)
    for index in range(count):
        smoothed[index] = padded[index : index + kernel].mean(axis=0)
    return smoothed


def jitter_rms(raw: np.ndarray, smooth: np.ndarray) -> float:
    delta = np.asarray(raw, dtype=np.float64) - np.asarray(smooth, dtype=np.float64)
    return float(np.sqrt(np.mean(delta**2)))


def estimate_homographies(frames: list[np.ndarray], masks: list[np.ndarray]) -> HomographyTrack:
    """Place a fixed-shape quad every frame, anchored to that frame's most reliable edge.

    Optical flow inside the mask cannot see a blank door or wall, which is the
    surface a poster needs, so SAM's per-frame outline is the only signal. But
    something passing in front of the surface (a table, a person) makes SAM's
    visible outline shrink on the occluded side; refitting a quad to that
    shrunken outline reads as the surface itself moving. Instead, the quad's
    shape is fixed to its most complete frame, and only the two corners on its
    most stable edge are used to slide and spin that fixed shape each frame.
    """
    if len(frames) != len(masks) or not frames:
        raise PipelineError("Frames and surface masks are not aligned")
    raw = np.full((len(frames), 4, 2), np.nan, dtype=np.float64)
    valid = np.zeros(len(frames), dtype=bool)
    previous: np.ndarray | None = None
    observed = np.full((len(frames), 4, 2), np.nan, dtype=np.float64)
    for index, mask in enumerate(masks):
        if int((mask > 0).sum()) < 16:
            continue
        try:
            quad = quad_from_mask(mask).astype(np.float64)
        except PipelineError:
            continue
        if previous is not None:
            quad = _lock_order(quad, previous)
        observed[index] = quad
        previous = quad
    if not np.isfinite(observed).any():
        raise PipelineError("No frame produced a stable homography")
    reference = _reference_quad(masks, observed)
    anchor = _stable_edge(observed)
    for index in range(len(frames)):
        quad = observed[index]
        if not np.isfinite(quad).all():
            continue
        placed = _place_reference(reference, anchor, quad[list(anchor)])
        if placed is None:
            continue
        raw[index] = placed
        valid[index] = True
    if not valid.any():
        raise PipelineError("No frame produced a stable homography")
    filled = _interpolate(raw)
    smooth = smooth_tracks(filled, window=9)
    unit = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    homographies = np.empty((len(frames), 3, 3), dtype=np.float64)
    for index in range(len(frames)):
        homographies[index] = cv2.getPerspectiveTransform(unit, smooth[index].astype(np.float32))
    return HomographyTrack(
        corners_raw=filled,
        corners_smooth=smooth,
        homographies=homographies,
        jitter_rms=_playback_jitter(smooth),
        valid=valid,
    )


def quad_from_mask(mask: np.ndarray) -> np.ndarray:
    contours = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_list = contours[0]
    if not contour_list:
        raise PipelineError("Surface mask is empty")
    contour = max(contour_list, key=cv2.contourArea)
    mask_area = float(cv2.contourArea(contour))
    perimeter = cv2.arcLength(contour, True)
    for scale in (0.01, 0.02, 0.04, 0.08):
        approx = cv2.approxPolyDP(contour, scale * perimeter, True)
        if len(approx) != 4:
            continue
        quad = order_corners(approx.reshape(4, 2))
        if _quad_is_solid(quad, mask_area):
            return quad
    box = cv2.boxPoints(cv2.minAreaRect(contour))
    return order_corners(box)


def order_corners(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]
    return np.stack(
        [
            pts[np.argmin(sums)],
            pts[np.argmax(diff)],
            pts[np.argmax(sums)],
            pts[np.argmin(diff)],
        ]
    ).astype(np.float32)


def _quad_is_solid(quad: np.ndarray, mask_area: float) -> bool:
    area = abs(float(cv2.contourArea(quad.astype(np.float32))))
    if area < max(16.0, 0.25 * mask_area):
        return False
    edges = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
    return bool(float(edges.min()) > 2.0)


_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))


def _reference_quad(masks: list[np.ndarray], observed: np.ndarray) -> np.ndarray:
    """The door's fixed shape, taken from its most fully visible frame."""
    areas = np.array([int((mask > 0).sum()) for mask in masks], dtype=np.float64)
    areas[~np.isfinite(observed).all(axis=(1, 2))] = -1.0
    best = int(np.argmax(areas))
    if areas[best] < 0:
        raise PipelineError("No frame produced a stable homography")
    return observed[best]


def _stable_edge(observed: np.ndarray) -> tuple[int, int]:
    """The pair of adjacent corners that moves least frame to frame.

    Occlusion only shrinks the mask on the side it covers, so the opposite
    edge keeps tracking the real surface while that edge's corners swing.
    """
    best_edge = _EDGES[0]
    best_score = float("inf")
    for a, b in _EDGES:
        pair = observed[:, (a, b)]
        finite = np.isfinite(pair).all(axis=(1, 2))
        if finite.sum() < 2:
            continue
        step = np.linalg.norm(np.diff(pair[finite], axis=0), axis=-1)
        if step.size == 0:
            continue
        score = float(np.median(step))
        if score < best_score:
            best_score = score
            best_edge = (a, b)
    return best_edge


def _similarity_transform(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Translation + uniform scale + rotation mapping the two ``src`` points to ``dst``."""
    src_vector = src[1] - src[0]
    dst_vector = dst[1] - dst[0]
    src_length = float(np.hypot(*src_vector))
    if src_length < 1e-6:
        return None
    scale = float(np.hypot(*dst_vector)) / src_length
    angle = math.atan2(dst_vector[1], dst_vector[0]) - math.atan2(src_vector[1], src_vector[0])
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    matrix = scale * np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)
    translation = dst[0] - matrix @ src[0]
    return matrix, translation


def _place_reference(reference: np.ndarray, anchor: tuple[int, int], anchor_points: np.ndarray) -> np.ndarray | None:
    """Slide and spin the fixed reference quad so its anchor edge lands on ``anchor_points``."""
    src = reference[list(anchor)]
    fit = _similarity_transform(src, anchor_points)
    if fit is None:
        return None
    matrix, translation = fit
    return reference @ matrix.T + translation


def _lock_order(quad: np.ndarray, previous: np.ndarray) -> np.ndarray:
    best = quad
    best_cost = float("inf")
    for direction in (quad, quad[::-1]):
        for shift in range(4):
            rolled = np.roll(direction, shift, axis=0)
            cost = float(np.linalg.norm(rolled - previous))
            if cost < best_cost:
                best = rolled
                best_cost = cost
    return best


def _playback_jitter(smooth: np.ndarray) -> float:
    if len(smooth) < 3:
        return 0.0
    accel = np.diff(np.asarray(smooth, dtype=np.float64), n=2, axis=0)
    return float(np.sqrt(np.mean(accel**2)))


def _interpolate(corners: np.ndarray) -> np.ndarray:
    filled = corners.copy()
    index = np.arange(len(filled))
    for corner in range(4):
        for axis in range(2):
            column = filled[:, corner, axis]
            good = np.isfinite(column)
            if not good.any():
                raise PipelineError("No usable frame for the surface")
            if not good.all():
                filled[:, corner, axis] = np.interp(index, index[good], column[good])
    return filled
