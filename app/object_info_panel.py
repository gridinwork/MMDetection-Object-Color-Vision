"""Details of the object selected in the live view."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget


class ObjectInfoPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.summary = QLabel("Click an object in the live view.")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.summary)
        row = QHBoxLayout()
        self.lock_button = QPushButton("LOCK OBJECT")
        self.save_button = QPushButton("SAVE OBJECT")
        self.mask_button = QPushButton("SAVE MASKED OBJECT")
        row.addWidget(self.lock_button)
        row.addWidget(self.save_button)
        row.addWidget(self.mask_button)
        layout.addLayout(row)
        self.save_button.setEnabled(False)
        self.mask_button.setEnabled(False)
        self.lock_button.setEnabled(False)

    def show_empty(self) -> None:
        self.summary.setText("Click an object in the live view.")
        self.save_button.setEnabled(False)
        self.mask_button.setEnabled(False)
        self.lock_button.setEnabled(False)

    def show_object(self, obj: dict, locked: bool) -> None:
        x1, y1, x2, y2 = [float(v) for v in obj["bbox"]]
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        track = obj.get("track_id")
        lines = [
            f"ID: {track if track is not None else '—'}",
            f"Class: {obj.get('class_name', '')}",
            f"Confidence: {float(obj.get('score', 0.0)):.2f}",
            "Bounding Box:",
            f"x: {x1:.0f}",
            f"y: {y1:.0f}",
            f"width: {width:.0f}",
            f"height: {height:.0f}",
            f"Dominant Color: {obj.get('color_name') or '—'}",
        ]
        if obj.get("rgb"):
            lines.append("RGB: " + ", ".join(str(int(v)) for v in obj["rgb"]))
        if obj.get("hsv"):
            lines.append("HSV: " + ", ".join(str(int(v)) for v in obj["hsv"]))
        if obj.get("lab"):
            lines.append("LAB: " + ", ".join(str(int(v)) for v in obj["lab"]))
        lines.append(f"Tracked: {'YES' if obj.get('tracked') else 'NO'}")
        lines.append(f"Age: {int(obj.get('age') or 0)} frames")
        if obj.get("motion"):
            vx, vy = obj.get("velocity") or (0.0, 0.0)
            lines.append(f"Motion: {obj['motion']}")
            lines.append(f"Velocity: {vx:.2f}, {vy:.2f} px/frame")
        self.summary.setText("\n".join(lines))
        self.save_button.setEnabled(True)
        self.mask_button.setEnabled(obj.get("mask") is not None)
        self.lock_button.setEnabled(bool(obj.get("tracked")) and track is not None)
        self.lock_button.setText("UNLOCK" if locked else "LOCK OBJECT")
