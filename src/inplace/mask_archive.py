from __future__ import annotations

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np


def save_mask_archive(
    path: Path,
    masks: list[np.ndarray],
    *,
    fps: float,
) -> None:
    if not masks:
        raise ValueError("mask archive needs at least one frame")
    height, width = masks[0].shape[:2]
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "meta.json",
            json.dumps({"frame_count": len(masks), "width": width, "height": height, "fps": fps}),
        )
        for index, mask in enumerate(masks):
            if int(mask.max()) == 0:
                continue
            ok, encoded = cv2.imencode(".png", mask)
            if not ok:
                raise ValueError(f"could not encode mask {index}")
            archive.writestr(f"{index}.png", encoded.tobytes())


class MaskArchive:
    def __init__(self, path: Path) -> None:
        self._zip = zipfile.ZipFile(path)
        meta = json.loads(self._zip.read("meta.json"))
        self.frame_count = int(meta["frame_count"])
        self.width = int(meta["width"])
        self.height = int(meta["height"])
        self.fps = float(meta["fps"])
        self._names = set(self._zip.namelist())

    def frame(self, index: int) -> np.ndarray:
        name = f"{index}.png"
        if name not in self._names:
            return np.zeros((self.height, self.width), dtype=np.uint8)
        encoded = np.frombuffer(self._zip.read(name), dtype=np.uint8)
        mask = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros((self.height, self.width), dtype=np.uint8)
        return mask

    def close(self) -> None:
        self._zip.close()
