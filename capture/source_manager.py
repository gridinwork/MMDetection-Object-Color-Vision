"""Switch the active frame source."""

from __future__ import annotations

from capture.camera_capture import CameraCapture
from capture.image_source import ImageSource
from capture.video_capture import VideoCaptureSource


class SourceManager:
    def __init__(self):
        self.kind = ""
        self.camera = CameraCapture()
        self.video = VideoCaptureSource()
        self.image = ImageSource()

    def open(self, kind: str, **kwargs) -> None:
        self.close()
        self.kind = kind
        if kind == "webcam":
            self.camera.open(int(kwargs["index"]), int(kwargs["width"]), int(kwargs["height"]))
        elif kind == "video":
            self.video.open(kwargs["path"])
        elif kind == "image":
            self.image.open(kwargs["path"])
        else:
            raise RuntimeError(f"Unknown source: {kind}")

    def read(self):
        if self.kind == "webcam":
            return self.camera.read()
        if self.kind == "video":
            return self.video.read()
        if self.kind == "image":
            return self.image.read()
        return False, None, {}

    def seek_ratio(self, ratio: float) -> None:
        if self.kind == "video":
            self.video.seek_ratio(ratio)

    @property
    def video_fps(self) -> float:
        if self.kind == "video":
            return self.video.fps
        return 0.0

    def close(self) -> None:
        self.camera.close()
        self.video.close()
        self.image.close()
        self.kind = ""
