from __future__ import annotations

import math

import cv2
import numpy as np


def stamp_sponsored(frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    x0, y0 = 8, height - 28
    x1, y1 = min(width - 8, x0 + 118), height - 8
    if y0 < 0 or x1 <= x0:
        return
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
    cv2.putText(
        frame,
        "Sponsored",
        (x0 + 6, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_8,
    )


def composite_frame(
    frame: np.ndarray,
    creative: np.ndarray,
    surface: np.ndarray,
    occluder: np.ndarray,
    corners: np.ndarray,
    motion: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    height, width = frame.shape[:2]
    creative_h, creative_w = creative.shape[:2]
    source = np.array(
        [[0, 0], [creative_w - 1, 0], [creative_w - 1, creative_h - 1], [0, creative_h - 1]],
        dtype=np.float32,
    )
    destination = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    homography = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(creative, homography, (width, height))
    visible = ((surface > 0) & (occluder == 0)).astype(np.float32)
    alpha = cv2.GaussianBlur(visible, (0, 0), sigmaX=1.5)
    alpha[occluder > 0] = 0
    alpha = np.clip(alpha, 0, 1)
    gain = _light_gain(frame, surface)
    shaded = np.clip(warped.astype(np.float32) * gain[:, :, None], 0, 255)
    distance = math.hypot(motion[0], motion[1])
    if distance >= 0.75:
        shaded, alpha = _motion_blur(shaded, alpha, motion)
        alpha[occluder > 0] = 0
    blend = alpha[:, :, None]
    merged = frame.astype(np.float32) * (1.0 - blend) + shaded * blend
    out = np.clip(merged, 0, 255).astype(np.uint8)
    stamp_sponsored(out)
    return out


def _light_gain(frame: np.ndarray, surface: np.ndarray) -> np.ndarray:
    """Stretch the scene's soft light so it reads on the poster.

    A white door only varies by a few percent. Dividing by the median leaves
    that falloff almost invisible, so the poster looks unlit.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    covered = surface > 0
    reference = float(np.median(gray[covered])) if covered.any() else float(np.median(gray))
    soft = cv2.GaussianBlur(gray, (0, 0), sigmaX=28)
    ratio = soft / max(reference, 1.0)
    return np.clip(1.0 + (ratio - 1.0) * 2.4, 0.55, 1.45)


def _motion_blur(
    image: np.ndarray,
    alpha: np.ndarray,
    motion: tuple[float, float],
    steps: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = alpha.shape[:2]
    image_acc = np.zeros_like(image, dtype=np.float32)
    alpha_acc = np.zeros_like(alpha, dtype=np.float32)
    for step in range(steps):
        shift = (step / (steps - 1) - 0.5) if steps > 1 else 0.0
        matrix = np.array(
            [[1.0, 0.0, motion[0] * shift], [0.0, 1.0, motion[1] * shift]],
            dtype=np.float32,
        )
        image_acc += cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR)
        alpha_acc += cv2.warpAffine(alpha, matrix, (width, height), flags=cv2.INTER_LINEAR)
    return image_acc / steps, alpha_acc / steps
