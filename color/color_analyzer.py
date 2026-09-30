"""Measure object color from mask pixels or the inner bounding box."""

from __future__ import annotations

import cv2
import numpy as np

from color.color_classifier import ColorClassifier
from color.dominant_colors import kmeans_colors


def clip_box(box, width: int, height: int):
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(width - 1.0, x1))
    y1 = max(0.0, min(height - 1.0, y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return int(x1), int(y1), int(x2), int(y2)


def _sample_pixels(image: np.ndarray, bbox, mask, max_samples: int = 1800) -> np.ndarray | None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = clip_box(bbox, width, height)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = image[y1:y2, x1:x2]
    if mask is not None:
        mask_arr = mask
        if mask_arr.shape[:2] != image.shape[:2]:
            mask_arr = cv2.resize(
                mask_arr.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        region = mask_arr[y1:y2, x1:x2] > 0
        if int(region.sum()) > 80:
            eroded = cv2.erode(region.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1) > 0
            if int(eroded.sum()) > 30:
                region = eroded
        pixels = crop[region]
    else:
        box_h, box_w = crop.shape[:2]
        margin_x = int(box_w * 0.25)
        margin_y = int(box_h * 0.25)
        inner = crop[margin_y: box_h - margin_y or box_h, margin_x: box_w - margin_x or box_w]
        if inner.size < 12:
            inner = crop
        pixels = inner.reshape(-1, 3)
    if pixels is None or len(pixels) == 0:
        return None
    if len(pixels) > max_samples:
        indexes = np.linspace(0, len(pixels) - 1, max_samples).astype(np.int32)
        pixels = pixels[indexes]
    return np.ascontiguousarray(pixels)


def _bgr_to_metrics(bgr_pixel: np.ndarray) -> tuple[tuple, tuple, tuple]:
    pixel = np.uint8([[np.clip(bgr_pixel, 0, 255).astype(np.uint8)]])
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
    lab = cv2.cvtColor(pixel, cv2.COLOR_BGR2LAB)[0, 0]
    blue, green, red = [int(v) for v in pixel[0, 0]]
    rgb = (red, green, blue)
    hsv_t = (int(hsv[0]), int(hsv[1]), int(hsv[2]))
    lab_t = (int(lab[0]), int(lab[1]), int(lab[2]))
    return rgb, hsv_t, lab_t


class ColorAnalyzer:
    def __init__(self, classifier: ColorClassifier | None = None):
        self.classifier = classifier or ColorClassifier()

    def measure(self, image: np.ndarray, bbox, mask=None, palette: bool = False) -> dict | None:
        pixels = _sample_pixels(image, bbox, mask)
        if pixels is None:
            return None
        lab_pixels = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
        median_lab = np.median(lab_pixels, axis=0)
        median_bgr = cv2.cvtColor(np.uint8([[median_lab]]), cv2.COLOR_LAB2BGR)[0, 0]
        rgb, hsv, lab = _bgr_to_metrics(median_bgr)
        name = self.classifier.classify(hsv, lab)
        swatches = []
        if palette:
            merged: dict[str, dict] = {}
            for cluster in kmeans_colors(pixels, k=3, iterations=8):
                cluster_rgb, cluster_hsv, cluster_lab = _bgr_to_metrics(cluster["bgr"])
                cluster_name = self.classifier.classify(cluster_hsv, cluster_lab)
                current = merged.get(cluster_name)
                if current is None:
                    merged[cluster_name] = {
                        "name": cluster_name,
                        "rgb": cluster_rgb,
                        "ratio": cluster["ratio"],
                    }
                else:
                    current["ratio"] += cluster["ratio"]
            swatches = sorted(merged.values(), key=lambda item: item["ratio"], reverse=True)
        return {
            "name": name,
            "rgb": rgb,
            "hsv": hsv,
            "lab": lab,
            "palette": swatches,
        }
