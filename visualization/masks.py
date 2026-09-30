"""Semi-transparent instance masks."""

from __future__ import annotations

import cv2
import numpy as np


def prepare_mask(mask, width: int, height: int) -> np.ndarray | None:
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.shape[:2] != (height, width):
        arr = cv2.resize(arr.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return arr > 0


def blend_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> None:
    if mask is None or not np.any(mask):
        return
    overlay = image.copy()
    overlay[mask] = color
    blended = cv2.addWeighted(image, 1.0 - alpha, overlay, alpha, 0.0)
    image[mask] = blended[mask]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(image, contours, -1, color, 2, lineType=cv2.LINE_AA)
