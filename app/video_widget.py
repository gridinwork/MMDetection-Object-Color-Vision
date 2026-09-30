"""Live view drawn inside Qt. Clicks map back to image pixels."""

from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget


def bgr_to_qimage(image: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    height, width, _channels = rgb.shape
    qimage = QImage(rgb.data, width, height, width * 3, QImage.Format.Format_RGB888)
    return qimage.copy()


class VideoWidget(QWidget):
    clicked = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 360)
        self._image: QImage | None = None
        self._frame_size = (0, 0)
        self._dest = QRect()
        self._message = "LIVE VIEW\nPress START"

    def set_message(self, text: str) -> None:
        self._message = text
        self._image = None
        self.update()

    def set_frame(self, image_bgr: np.ndarray) -> None:
        if image_bgr is None or image_bgr.size == 0:
            return
        self._frame_size = (image_bgr.shape[1], image_bgr.shape[0])
        self._image = bgr_to_qimage(image_bgr)
        self._message = ""
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if self._image is None or self._image.isNull():
            painter.setPen(QPen(Qt.GlobalColor.white))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._message or "LIVE VIEW")
            self._dest = QRect()
            return
        frame_w, frame_h = self._frame_size
        if frame_w <= 0 or frame_h <= 0:
            return
        scale = min(self.width() / frame_w, self.height() / frame_h)
        draw_w = max(1, int(frame_w * scale))
        draw_h = max(1, int(frame_h * scale))
        left = (self.width() - draw_w) // 2
        top = (self.height() - draw_h) // 2
        self._dest = QRect(left, top, draw_w, draw_h)
        painter.drawImage(self._dest, self._image)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._dest.isNull():
            return
        point: QPoint = event.position().toPoint()
        if not self._dest.contains(point):
            return
        frame_w, frame_h = self._frame_size
        x = (point.x() - self._dest.x()) / max(1, self._dest.width()) * frame_w
        y = (point.y() - self._dest.y()) / max(1, self._dest.height()) * frame_h
        self.clicked.emit(float(x), float(y))
