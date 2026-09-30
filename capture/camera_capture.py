"""USB webcam capture through OpenCV."""

from __future__ import annotations

import threading

import cv2

from utils.logger import get_logger

log = get_logger("camera")

# One opener at a time. A second VideoCapture on the same index makes
# DirectShow throw a C++ exception and can abort the process.
_LOCK = threading.RLock()


def _quiet():
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        return


def list_cameras(max_index: int = 3) -> list[int]:
    found = []
    _quiet()
    with _LOCK:
        for index in range(max_index + 1):
            capture = _open_backend(index, cv2.CAP_DSHOW)
            if capture is None:
                continue
            try:
                ok, _frame = capture.read()
            except cv2.error:
                ok = False
            capture.release()
            if ok:
                found.append(index)
                log.info("Camera %s is available", index)
    return found


def open_camera(index: int, width: int, height: int):
    """Open one camera at MJPG and the requested size.

    DirectShow applies 1280x720 in a few seconds on this PC.
    Media Foundation renegotiates the same mode for about half a minute,
    so it is only the fallback.
    """
    _quiet()
    with _LOCK:
        capture = _open_sized(index, cv2.CAP_DSHOW, width, height)
        if capture is None:
            log.info("DirectShow did not open camera %s, trying Media Foundation", index)
            capture = _open_sized(index, cv2.CAP_MSMF, width, height)
        return capture


def _open_sized(index: int, backend: int, width: int, height: int):
    capture = _open_backend(index, backend)
    if capture is None:
        return None
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    return capture


def _open_backend(index: int, backend: int):
    try:
        capture = cv2.VideoCapture(int(index), backend)
    except cv2.error:
        log.warning("Camera %s backend %s failed", index, backend)
        return None
    if capture is not None and capture.isOpened():
        return capture
    if capture is not None:
        capture.release()
    return None


class CameraCapture:
    def __init__(self):
        self.capture = None
        self.index = 0
        self.frame_index = 0

    def open(self, index: int, width: int, height: int) -> None:
        self.close()
        log.info("Opening camera %s at %sx%s", index, width, height)
        capture = open_camera(index, width, height)
        if capture is None:
            raise RuntimeError(f"Camera {index} is not available.")
        self.capture = capture
        self.index = int(index)
        self.frame_index = 0
        actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        log.info("Camera %s started at %sx%s", index, actual_w, actual_h)

    def read(self):
        if self.capture is None:
            return False, None, {}
        try:
            ok, frame = self.capture.read()
        except cv2.error:
            log.exception("Camera %s read failed", self.index)
            return False, None, {}
        if not ok or frame is None or getattr(frame, "size", 0) == 0:
            return False, None, {}
        self.frame_index += 1
        fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        meta = {
            "frame_index": self.frame_index,
            "time_sec": 0.0,
            "duration_sec": 0.0,
            "fps": fps,
            "source": "webcam",
        }
        return True, frame, meta

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            log.info("Camera %s stopped", self.index)
        self.capture = None
