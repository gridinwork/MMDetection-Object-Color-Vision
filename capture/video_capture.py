"""Video file capture."""

from __future__ import annotations

from pathlib import Path

import cv2

from utils.logger import get_logger

log = get_logger("video")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


class VideoCaptureSource:
    def __init__(self):
        self.capture = None
        self.path = ""
        self.fps = 30.0
        self.frame_count = 0

    def open(self, path: str) -> None:
        self.close()
        file_path = Path(path)
        if file_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise RuntimeError("Supported video files: mp4, avi, mov, mkv.")
        capture = cv2.VideoCapture(str(file_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video: {file_path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps if fps > 1.0 else 30.0
        self.frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.capture = capture
        self.path = str(file_path)
        log.info("Video opened %s fps %.2f frames %s", file_path.name, self.fps, self.frame_count)

    def read(self):
        if self.capture is None:
            return False, None, {}
        ok, frame = self.capture.read()
        if not ok or frame is None:
            return False, None, self.meta()
        return True, frame, self.meta()

    def meta(self) -> dict:
        position = 0.0
        if self.capture is not None:
            position = float(self.capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        duration = (self.frame_count / self.fps) if self.fps else 0.0
        index = 0
        if self.capture is not None:
            index = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES) or 0)
        return {
            "frame_index": index,
            "time_sec": position,
            "duration_sec": duration,
            "fps": self.fps,
            "source": "video",
        }

    def seek_ratio(self, ratio: float) -> None:
        if self.capture is None or self.frame_count <= 0:
            return
        ratio = min(1.0, max(0.0, float(ratio)))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(ratio * max(0, self.frame_count - 1)))

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            log.info("Video closed")
        self.capture = None
