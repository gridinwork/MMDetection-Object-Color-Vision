"""Tracker and color timing settings."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QSpinBox


class SettingsDialog(QDialog):
    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        layout = QFormLayout(self)
        self.lost = QSpinBox()
        self.lost.setRange(1, 120)
        self.lost.setValue(int(settings.get("lost_frames_timeout") or 20))
        self.color_interval = QSpinBox()
        self.color_interval.setRange(1, 60)
        self.color_interval.setValue(int(settings.get("color_interval") or 5))
        self.trail = QSpinBox()
        self.trail.setRange(2, 120)
        self.trail.setValue(int(settings.get("trail_length") or 30))
        layout.addRow("Lost frames timeout", self.lost)
        layout.addRow("Color palette interval (frames)", self.color_interval)
        layout.addRow("Trail length", self.trail)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def values(self) -> dict:
        return {
            "lost_frames_timeout": int(self.lost.value()),
            "color_interval": int(self.color_interval.value()),
            "trail_length": int(self.trail.value()),
        }
