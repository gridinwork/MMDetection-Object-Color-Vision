"""Single-image source."""

from __future__ import annotations

from pathlib import Path

import cv2

from utils.logger import get_logger

log = get_logger("image")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class ImageSource:
    def __init__(self):
        self.image = None
        self.path = ""

    def open(self, path: str) -> None:
        file_path = Path(path)
        if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise RuntimeError("Supported images: jpg, jpeg, png, webp.")
        image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {file_path}")
        self.image = image
        self.path = str(file_path)
        log.info("Image opened %s %sx%s", file_path.name, image.shape[1], image.shape[0])

    def read(self):
        if self.image is None:
            return False, None, {}
        meta = {
            "frame_index": 0,
            "time_sec": 0.0,
            "duration_sec": 0.0,
            "fps": 0.0,
            "source": "image",
        }
        return True, self.image.copy(), meta

    def close(self) -> None:
        self.image = None
