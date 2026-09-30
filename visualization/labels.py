"""Text labels drawn above a detection."""

from __future__ import annotations

import cv2
import numpy as np


def _ascii(text: str) -> str:
    return text.encode("ascii", "replace").decode("ascii")


def _hits(rect, occupied) -> bool:
    left, top, right, bottom = rect
    for other in occupied:
        if right <= other[0] or other[2] <= left or bottom <= other[1] or other[3] <= top:
            continue
        return True
    return False


def draw_label_block(
    image: np.ndarray,
    anchor: tuple[int, int],
    lines: list[tuple[str, tuple[int, int, int] | None]],
    font_scale: float,
    occupied: list | None = None,
) -> None:
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    pad = 4
    sizes = []
    for text, _swatch in lines:
        (tw, th), baseline = cv2.getTextSize(_ascii(text), font, font_scale, thickness)
        sizes.append((tw, th, baseline))
    block_w = max(item[0] for item in sizes) + pad * 2 + 18
    line_h = max(item[1] + item[2] for item in sizes) + 4
    block_h = line_h * len(lines) + pad
    x, y = anchor
    height, width = image.shape[:2]
    if y - block_h < 0:
        y = min(height - 1, y + block_h)
    x = max(0, min(x, width - block_w - 1))
    top = max(0, y - block_h)
    bottom = min(height, top + block_h)
    left = x
    right = min(width, left + block_w)
    if occupied is not None and bottom > top and right > left:
        for _step in range(14):
            rect = (left, top, right, bottom)
            if not _hits(rect, occupied):
                break
            top = min(height - (bottom - top), top + (bottom - top) + 2)
            bottom = min(height, top + block_h)
            if bottom >= height and _hits((left, top, right, bottom), occupied):
                left = min(width - (right - left), left + 24)
                right = min(width, left + block_w)
        occupied.append((left, top, right, bottom))
    if bottom <= top or right <= left:
        return
    roi = image[top:bottom, left:right]
    image[top:bottom, left:right] = (roi.astype(np.float32) * 0.35).astype(np.uint8)
    cursor = top + pad
    for (text, swatch), (tw, th, baseline) in zip(lines, sizes):
        text_x = left + pad
        text_y = min(height - 2, cursor + th)
        if swatch is not None:
            x1, y1 = text_x, cursor + 2
            x2, y2 = min(width - 1, text_x + 12), min(height - 1, cursor + 14)
            cv2.rectangle(image, (x1, y1), (x2, y2), swatch, thickness=-1)
            text_x += 16
        cv2.putText(
            image,
            _ascii(text),
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            lineType=cv2.LINE_AA,
        )
        cursor += line_h
