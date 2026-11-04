import cv2
import numpy as np

from inplace.constants import JITTER_BUDGET
from inplace.stages.warp import estimate_homographies, jitter_rms, smooth_tracks


def test_smooth_track_residual_is_the_jitter() -> None:
    frames = 21
    base = np.zeros((frames, 4, 2), dtype=np.float64)
    for i in range(frames):
        base[i, :, 0] = i * 2
        base[i, :, 1] = 10
    noisy = base.copy()
    noisy[::2] += 8
    smooth = smooth_tracks(noisy, window=5)
    assert jitter_rms(base, smooth_tracks(base, window=5)) < 0.05
    assert jitter_rms(noisy, smooth) > JITTER_BUDGET


def test_a_blank_surface_is_tracked_from_its_mask() -> None:
    frames: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    truth = []
    for index in range(12):
        frame = np.full((120, 160, 3), 30, dtype=np.uint8)
        mask = np.zeros((120, 160), dtype=np.uint8)
        x, y = 20 + index * 2, 15 + index
        mask[y : y + 50, x : x + 40] = 255
        frame[mask > 0] = 240
        frames.append(frame)
        masks.append(mask)
        truth.append(
            np.array(
                [[x, y], [x + 39, y], [x + 39, y + 49], [x, y + 49]],
                dtype=np.float32,
            )
        )
    track = estimate_homographies(frames, masks)
    assert track.valid.all()
    error = np.linalg.norm(track.corners_smooth - np.stack(truth), axis=-1).mean()
    assert error < 4.0
    assert track.jitter_rms < JITTER_BUDGET


def test_estimate_homographies_follows_a_moving_textured_quad() -> None:
    frames, masks, truth = _moving_quad(n=10)
    track = estimate_homographies(frames, masks)
    assert track.jitter_rms < JITTER_BUDGET
    error = np.linalg.norm(track.corners_smooth - truth, axis=-1).mean()
    assert error < 4.0


def _moving_quad(n: int) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    height, width = 180, 240
    texture = np.zeros((80, 100, 3), dtype=np.uint8)
    texture[::8, :] = 255
    texture[:, ::8] = 255
    cv2.circle(texture, (30, 25), 8, (0, 0, 255), -1)
    cv2.circle(texture, (70, 50), 6, (0, 255, 0), -1)
    src = np.array([[0, 0], [99, 0], [99, 79], [0, 79]], dtype=np.float32)
    truth = []
    frames = []
    masks = []
    for i in range(n):
        shift_x = 20 + i * 3
        shift_y = 30 + i
        dest = np.array(
            [
                [shift_x, shift_y],
                [shift_x + 90, shift_y + 4],
                [shift_x + 84, shift_y + 70],
                [shift_x + 2, shift_y + 64],
            ],
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(src, dest)
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        warped = cv2.warpPerspective(texture, homography, (width, height))
        mask = cv2.warpPerspective(np.full(texture.shape[:2], 255, np.uint8), homography, (width, height))
        mask = np.where(mask > 127, 255, 0).astype(np.uint8)
        frame[mask > 0] = warped[mask > 0]
        frames.append(frame)
        masks.append(mask)
        truth.append(dest)
    return frames, masks, np.stack(truth)
