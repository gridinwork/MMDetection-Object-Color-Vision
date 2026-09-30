"""Draw boxes, masks, labels, trails, and the locked target."""

from __future__ import annotations

import cv2
import numpy as np

from visualization.colors import color_for_key
from visualization.labels import draw_label_block
from visualization.masks import blend_mask, prepare_mask


def _disabled(cfg) -> set[str]:
    return {str(name) for name in (cfg.get("disabled_classes") or [])}


def _swatch_bgr(rgb) -> tuple[int, int, int] | None:
    if not rgb or len(rgb) < 3:
        return None
    red, green, blue = [int(v) for v in rgb[:3]]
    return blue, green, red


def render(image: np.ndarray, objects: list[dict], cfg: dict) -> np.ndarray:
    canvas = image.copy()
    height, width = canvas.shape[:2]
    font_scale = max(0.45, min(0.75, width / 1700.0))
    disabled = _disabled(cfg)
    show_masks = bool(cfg.get("segmentation"))
    show_boxes = bool(cfg.get("detection", True))
    show_labels = bool(cfg.get("labels", True))
    show_conf = bool(cfg.get("show_confidence", True))
    show_id = bool(cfg.get("show_id", True))
    show_color = bool(cfg.get("color"))
    show_detail = bool(cfg.get("detailed_color"))
    show_palette = bool(cfg.get("palette"))
    show_trails = bool(cfg.get("trails"))
    locked_id = cfg.get("locked_id")
    selected_id = cfg.get("selected_id")
    locked_caption = cfg.get("locked_caption") or ""

    visible_locked = False
    occupied = []
    for obj in objects:
        if obj.get("class_name") in disabled:
            continue
        key = obj.get("track_id")
        identity = obj.get("select_key", key)
        color_key = key if isinstance(key, int) else abs(hash(str(identity))) % 10_000
        color = color_for_key(color_key)
        x1, y1, x2, y2 = [int(round(float(v))) for v in obj["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        if show_trails and obj.get("trail"):
            points = [(int(px), int(py)) for px, py in obj["trail"]]
            if len(points) >= 2:
                cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        if show_masks and obj.get("mask") is not None:
            mask = prepare_mask(obj["mask"], width, height)
            blend_mask(canvas, mask, color, alpha=0.45)
        thickness = 2
        if locked_id is not None and key is not None and int(key) == int(locked_id):
            thickness = 4
            visible_locked = True
        if selected_id is not None and identity is not None and str(identity) == str(selected_id):
            cv2.rectangle(canvas, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 1, cv2.LINE_AA)
        if show_boxes or show_masks:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        lines = []
        parts = []
        if show_id and key is not None:
            parts.append(f"#{int(key)}")
        if show_labels:
            parts.append(str(obj.get("class_name", "")).upper())
        if show_conf:
            parts.append(f"{float(obj.get('score', 0.0)) * 100:.0f}%")
        if parts:
            lines.append((" | ".join(parts), None))
        if show_color and obj.get("color_name"):
            lines.append((f"COLOR: {obj['color_name']}", _swatch_bgr(obj.get("rgb"))))
        if show_detail and obj.get("rgb"):
            red, green, blue = obj["rgb"]
            lines.append((f"RGB: {red}, {green}, {blue}", None))
            if obj.get("hsv"):
                lines.append(("HSV: " + ", ".join(str(int(v)) for v in obj["hsv"]), None))
            if obj.get("lab"):
                lines.append(("LAB: " + ", ".join(str(int(v)) for v in obj["lab"]), None))
        if show_palette:
            for swatch in obj.get("palette") or []:
                ratio = int(round(float(swatch.get("ratio", 0.0)) * 100))
                lines.append((f"{swatch.get('name', '')} {ratio}%", _swatch_bgr(swatch.get("rgb"))))
        if lines:
            draw_label_block(canvas, (x1, y1), lines, font_scale, occupied)

    if locked_caption:
        banner = locked_caption if visible_locked else f"TARGET LOST: {locked_caption}"
        cv2.rectangle(canvas, (8, 8), (min(width - 8, 8 + 18 * len(banner)), 42), (20, 20, 20), -1)
        cv2.putText(
            canvas,
            banner.encode("ascii", "replace").decode("ascii"),
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (80, 220, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas
