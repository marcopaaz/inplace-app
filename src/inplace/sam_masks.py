from __future__ import annotations

from typing import Literal, Never

import cv2
import numpy as np
from meta_sam_parser._errors import InvalidSegmentationMaskError
from meta_sam_parser._mask_codec import _decode_lossless_raster, _decode_raster
from meta_sam_parser._segmentation import (
    _API_HEADER_PATTERN,
    _RECORD_BOUNDARY,
    _RECORD_START_PATTERN,
    _TOKEN_PATTERN,
    _ApiToken,
    _integer_text,
    _read_api_record,
)

from inplace.errors import PipelineError

MediaKind = Literal["video", "image"]


def rasterize_sam(
    text: str,
    *,
    frame_count: int,
    width: int,
    height: int,
    media: MediaKind = "video",
    stride: int = 1,
) -> list[np.ndarray]:
    """Decode a SAM stream into one full-frame mask per kept source frame.

    Masks are box-local. Missing frame indices stay empty. A frame index past
    the clip is a hard failure, because the warp would no longer line up.
    ``stride`` keeps frame 0, ``stride``, ``2 * stride``, and so on. The public
    parser decodes every mask and then encodes it again to prove the payload;
    a phone clip has hundreds of masks, so this path decodes each kept frame once.
    """
    if frame_count < 1 or width < 1 or height < 1:
        raise PipelineError("SAM raster needs a positive frame count and size")
    if stride < 1:
        raise PipelineError("SAM frame stride must be at least 1")
    kept = (frame_count + stride - 1) // stride
    masks = [np.zeros((height, width), dtype=np.uint8) for _ in range(kept)]
    for frame_index, left, top, right, bottom, payload, mask_width, mask_height in _records(text, media):
        if frame_index < 0 or frame_index >= frame_count:
            raise PipelineError(f"SAM frame {frame_index} is outside the {frame_count}-frame clip")
        if frame_index % stride != 0:
            continue
        patch = _decode(payload, mask_width, mask_height)
        box_w = max(1, right - left)
        box_h = max(1, bottom - top)
        if patch.shape[1] != box_w or patch.shape[0] != box_h:
            patch = cv2.resize(patch, (box_w, box_h), interpolation=cv2.INTER_NEAREST)
        _blit(masks[frame_index // stride], patch, left, top)
    return masks


def _records(
    text: str,
    media: MediaKind,
) -> list[tuple[int, int, int, int, int, str, int, int]]:
    found: list[tuple[int, int, int, int, int, str, int, int]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        header = _API_HEADER_PATTERN.fullmatch(line)
        if header is None or not header.group(3):
            raise PipelineError("SAM returned a record that could not be read")
        frame_index = _integer_text(header.group(1))
        if frame_index is None or frame_index < 0:
            raise PipelineError("SAM frame references must be non-negative integers")
        _check_media(media, frame_index)
        body = header.group(3)
        position = 0
        while position < len(body):
            start = _RECORD_START_PATTERN.match(body, position)
            tokens: list[_ApiToken] = []
            end = position if start is None else start.end()
            token = None if start is None else _TOKEN_PATTERN.match(body, end)
            while token is not None:
                tokens.append(_ApiToken(name=token.group(1), body=token.group(2)))
                end = token.end()
                token = _TOKEN_PATTERN.match(body, end)
            record = (
                None
                if start is None or (end < len(body) and body[end] not in _RECORD_BOUNDARY)
                else _read_api_record(start.group(1), tokens, [])
            )
            if record is None:
                raise PipelineError("SAM returned a record that could not be read")
            coordinates = tuple(_integer_text(value) for value in record.box)
            if any(value is None for value in coordinates):
                raise PipelineError("SAM box coordinates are invalid")
            left, top, inclusive_right, inclusive_bottom, source_width, source_height = coordinates
            if (
                left is None
                or top is None
                or inclusive_right is None
                or inclusive_bottom is None
                or source_width is None
                or source_height is None
                or source_width <= 0
                or source_height <= 0
                or left < 0
                or top < 0
                or inclusive_right < left
                or inclusive_bottom < top
                or inclusive_right >= source_width
                or inclusive_bottom >= source_height
            ):
                raise PipelineError("SAM box coordinates are invalid")
            found.append(
                (
                    frame_index,
                    left,
                    top,
                    inclusive_right + 1,
                    inclusive_bottom + 1,
                    record.payload,
                    record.mask_width,
                    record.mask_height,
                )
            )
            position = end
    return found


def _check_media(media: MediaKind, frame_index: int) -> None:
    if media == "video":
        return
    if media == "image":
        if frame_index != 0:
            raise PipelineError("Image segmentation records require frame zero.")
        return
    unexpected: Never = media
    raise PipelineError(f"Unknown SAM media {unexpected}")


def _decode(payload: str, width: int, height: int) -> np.ndarray:
    try:
        if payload.startswith("~"):
            coverage = _decode_lossless_raster(payload, width, height)
            raw = bytes(value >= 129 for value in coverage)
        elif payload.startswith("!"):
            raw = _decode_raster(payload, width, height)
        else:
            raise PipelineError("SAM mask encoding is not supported")
    except InvalidSegmentationMaskError as exc:
        raise PipelineError(str(exc)) from exc
    raster = np.frombuffer(raw, dtype=np.uint8)
    if raster.size != width * height:
        raise PipelineError("SAM mask size does not match its box")
    small = raster.reshape(height, width)
    return np.where(small > 0, np.uint8(255), np.uint8(0))


def _blit(canvas: np.ndarray, patch: np.ndarray, left: int, top: int) -> None:
    frame_h, frame_w = canvas.shape[:2]
    patch_h, patch_w = patch.shape[:2]
    x0 = max(0, left)
    y0 = max(0, top)
    x1 = min(frame_w, left + patch_w)
    y1 = min(frame_h, top + patch_h)
    if x1 <= x0 or y1 <= y0:
        return
    region = canvas[y0:y1, x0:x1]
    incoming = patch[y0 - top : y0 - top + (y1 - y0), x0 - left : x0 - left + (x1 - x0)]
    np.maximum(region, incoming, out=region)
