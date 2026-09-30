"""Stable overlay colors for track IDs."""

from __future__ import annotations

import cv2
import numpy as np


def color_for_key(key: int) -> tuple[int, int, int]:
    hue = int((int(key) * 47) % 180)
    hsv = np.uint8([[[hue, 210, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])
