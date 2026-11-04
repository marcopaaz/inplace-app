import cv2
import numpy as np

from inplace.stages.composite import composite_frame, stamp_sponsored


def test_occluder_pixels_stay_on_the_original_frame() -> None:
    frame = np.full((200, 200, 3), 90, dtype=np.uint8)
    creative = np.zeros((40, 80, 3), dtype=np.uint8)
    creative[:, :] = (0, 0, 255)
    surface = np.zeros((200, 200), dtype=np.uint8)
    corners = np.array([[40, 50], [120, 50], [120, 90], [40, 90]], dtype=np.float32)
    cv2.fillConvexPoly(surface, corners.astype(np.int32), 255)
    occluder = np.zeros((200, 200), dtype=np.uint8)
    occluder[50:90, 40:80] = 255

    out = composite_frame(frame, creative, surface, occluder, corners, motion=(0.0, 0.0))

    assert np.array_equal(out[60:80, 50:70], frame[60:80, 50:70])
    visible = out[60:80, 90:110]
    assert visible[:, :, 2].mean() > frame[:, :, 2].mean() + 40


def test_soft_light_darkens_the_shadowed_side() -> None:
    frame = np.zeros((200, 240, 3), dtype=np.uint8)
    for x in range(240):
        frame[:, x] = int(190 + 30 * x / 239)
    creative = np.full((100, 120, 3), (40, 40, 210), dtype=np.uint8)
    surface = np.zeros((200, 240), dtype=np.uint8)
    corners = np.array([[40, 40], [200, 40], [200, 160], [40, 160]], dtype=np.float32)
    cv2.fillConvexPoly(surface, corners.astype(np.int32), 255)
    out = composite_frame(frame, creative, surface, np.zeros_like(surface), corners, motion=(0.0, 0.0))
    shadow = float(out[90:110, 55:75, 2].mean())
    light = float(out[90:110, 165:185, 2].mean())
    assert light > shadow * 1.12


def test_sponsored_badge_is_always_stamped() -> None:
    blank = np.zeros((80, 140, 3), dtype=np.uint8)
    stamped = blank.copy()
    stamp_sponsored(stamped)
    assert stamped.sum() > 0

    frame = np.full((200, 200, 3), 128, dtype=np.uint8)
    creative = np.full((20, 20, 3), 255, dtype=np.uint8)
    surface = np.zeros((200, 200), dtype=np.uint8)
    corners = np.array([[30, 30], [50, 30], [50, 50], [30, 50]], dtype=np.float32)
    cv2.fillConvexPoly(surface, corners.astype(np.int32), 255)
    out = composite_frame(frame, creative, surface, np.zeros_like(surface), corners, motion=(0.0, 0.0))
    badge = np.zeros_like(out)
    stamp_sponsored(badge)
    ys, xs = np.where(badge.max(axis=2) > 0)
    assert np.array_equal(out[ys, xs], badge[ys, xs])
