"""Capture and inference threads. The GUI only reads the latest result."""

from __future__ import annotations

import threading
import time

from PySide6.QtCore import QObject, QThread, Signal

from capture.camera_capture import list_cameras
from capture.source_manager import SourceManager
from inference.inference_manager import InferenceManager
from utils.logger import get_logger

log = get_logger("threads")


class LatestSlot:
    def __init__(self):
        self._lock = threading.Lock()
        self._item = None
        self.dropped = 0

    def push(self, item) -> None:
        with self._lock:
            if self._item is not None:
                self.dropped += 1
            self._item = item

    def pull(self):
        with self._lock:
            item = self._item
            self._item = None
            return item

    def stats(self) -> tuple[int, int]:
        with self._lock:
            waiting = 0 if self._item is None else 1
            return self.dropped, waiting

    def reset(self) -> None:
        with self._lock:
            self._item = None
            self.dropped = 0


class RuntimeConfig:
    def __init__(self):
        self._lock = threading.Lock()
        self._data = {}

    def replace(self, data: dict) -> None:
        with self._lock:
            self._data = dict(data)

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._data)


class FpsMeter:
    def __init__(self):
        self.value = 0.0
        self._count = 0
        self._mark = time.perf_counter()

    def tick(self) -> float:
        self._count += 1
        now = time.perf_counter()
        elapsed = now - self._mark
        if elapsed >= 0.5:
            self.value = self._count / elapsed
            self._count = 0
            self._mark = now
        return self.value


def infer_token(cfg: dict):
    return (
        cfg.get("model_id"),
        cfg.get("device"),
        round(float(cfg.get("confidence") or 0.4), 3),
        bool(cfg.get("detection")),
        bool(cfg.get("segmentation")),
        bool(cfg.get("tracking")),
        bool(cfg.get("color")),
        bool(cfg.get("palette")),
        round(float(cfg.get("processing_scale") or 1.0), 3),
        int(cfg.get("color_generation") or 0),
        int(cfg.get("lost_frames_timeout") or 20),
        int(cfg.get("color_interval") or 5),
        int(cfg.get("trail_length") or 30),
    )


def display_token(cfg: dict):
    disabled = tuple(sorted(str(name) for name in (cfg.get("disabled_classes") or [])))
    return (
        bool(cfg.get("labels", True)),
        bool(cfg.get("show_confidence", True)),
        bool(cfg.get("show_id", True)),
        bool(cfg.get("trails", False)),
        bool(cfg.get("detailed_color", False)),
        bool(cfg.get("palette", False)),
        bool(cfg.get("detection", True)),
        bool(cfg.get("segmentation", False)),
        disabled,
        cfg.get("locked_id"),
        str(cfg.get("selected_id")),
        str(cfg.get("locked_caption") or ""),
    )


class CaptureWorker(QThread):
    status = Signal(str)

    def __init__(self, slot: LatestSlot, settings: dict):
        super().__init__()
        self.slot = slot
        self.settings = dict(settings)
        self._stop = False
        self._paused = False
        self._lock = threading.Lock()
        self._seek = None
        self._playback = {"time": 0.0, "duration": 0.0, "fps": 0.0, "capture_fps": 0.0}
        self._fps = FpsMeter()

    def request_stop(self) -> None:
        self._stop = True

    def set_paused(self, paused: bool) -> None:
        self._paused = bool(paused)

    def request_seek(self, ratio: float) -> None:
        with self._lock:
            self._seek = float(ratio)

    def playback(self) -> dict:
        with self._lock:
            return dict(self._playback)

    def run(self) -> None:
        source = SourceManager()
        kind = self.settings.get("kind", "webcam")
        try:
            if kind == "webcam":
                self.status.emit(f"Opening camera {self.settings.get('index', 0)}...")
                source.open(
                    "webcam",
                    index=int(self.settings.get("index", 0)),
                    width=int(self.settings.get("width", 1280)),
                    height=int(self.settings.get("height", 720)),
                )
                self.status.emit(f"Camera {self.settings.get('index', 0)} started")
            elif kind == "video":
                source.open("video", path=self.settings["path"])
                self.status.emit("Video file started")
            else:
                source.open("image", path=self.settings["path"])
                self.status.emit("Image loaded")
            period = 0.0
            if kind == "video" and source.video_fps > 0:
                period = 1.0 / source.video_fps
            next_tick = time.perf_counter()
            image_sent = False
            while not self._stop:
                with self._lock:
                    seek = self._seek
                    self._seek = None
                if seek is not None:
                    source.seek_ratio(seek)
                if kind == "image" and image_sent:
                    time.sleep(0.05)
                    continue
                if self._paused and seek is None:
                    time.sleep(0.02)
                    continue
                ok, frame, meta = source.read()
                if not ok or frame is None:
                    if kind == "video":
                        self.status.emit("Video ended")
                        self._paused = True
                    else:
                        time.sleep(0.05)
                    continue
                meta = dict(meta)
                meta["capture_fps"] = self._fps.tick()
                dropped, waiting = self.slot.stats()
                meta["dropped"] = dropped
                meta["queue_size"] = waiting
                self.slot.push((frame.copy(), meta))
                with self._lock:
                    self._playback = {
                        "time": float(meta.get("time_sec") or 0.0),
                        "duration": float(meta.get("duration_sec") or 0.0),
                        "fps": float(meta.get("fps") or 0.0),
                        "capture_fps": float(meta["capture_fps"]),
                    }
                if kind == "image":
                    image_sent = True
                    continue
                if period > 0:
                    next_tick += period
                    delay = next_tick - time.perf_counter()
                    if delay > 0:
                        time.sleep(min(delay, 0.2))
                    else:
                        next_tick = time.perf_counter()
        except Exception as exc:
            log.exception("Capture failed")
            self.status.emit(str(exc))
        finally:
            source.close()
            if kind == "webcam":
                log.info("Camera stop requested")


class InferenceWorker(QThread):
    status = Signal(str)

    def __init__(self, frames: LatestSlot, outbox: LatestSlot, config: RuntimeConfig):
        super().__init__()
        self.frames = frames
        self.outbox = outbox
        self.config = config
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        manager = InferenceManager()
        manager.on_status = lambda text: self.status.emit(str(text))
        cache = None
        seen_infer = None
        seen_display = None
        fps = FpsMeter()
        try:
            while not self._stop:
                item = self.frames.pull()
                cfg = self.config.snapshot()
                itok = infer_token(cfg)
                dtok = display_token(cfg)
                if item is None and cache is not None and itok == seen_infer:
                    if dtok != seen_display:
                        output = manager.render_cached(cache, cfg)
                        output["inference_fps"] = fps.value
                        seen_display = dtok
                        self.outbox.push(output)
                    else:
                        time.sleep(0.01)
                    continue
                if item is None and (cache is None or itok == seen_infer):
                    time.sleep(0.01)
                    continue
                if item is None:
                    frame = cache["clean"]
                    meta = dict(cache.get("meta") or {})
                else:
                    frame, meta = item
                result = manager.analyze(frame, meta, cfg)
                seen_infer = itok
                seen_display = dtok
                cache = result
                result["inference_fps"] = fps.tick()
                if result.get("error"):
                    self.status.emit(result["error"])
                self.outbox.push(result)
        except Exception as exc:
            log.exception("Inference thread failed")
            self.status.emit(str(exc))
        finally:
            manager.shutdown()


class CameraScanThread(QThread):
    found = Signal(object)

    def run(self) -> None:
        try:
            cameras = list_cameras(3)
        except Exception:
            log.exception("Camera scan failed")
            cameras = []
        self.found.emit(cameras)


class VisionPipeline(QObject):
    status = Signal(str)
    idle = Signal()

    def __init__(self):
        super().__init__()
        self.config = RuntimeConfig()
        self.frames = LatestSlot()
        self.outbox = LatestSlot()
        self.capture: CaptureWorker | None = None
        self.infer: InferenceWorker | None = None
        self._pending = None

    def is_running(self) -> bool:
        capture_on = self.capture is not None and self.capture.isRunning()
        infer_on = self.infer is not None and self.infer.isRunning()
        return capture_on or infer_on

    def start(self, source_settings: dict) -> None:
        if self.is_running():
            self._pending = dict(source_settings)
            self.request_stop()
            self.status.emit("Restarting...")
            return
        self._launch(source_settings)

    def _launch(self, source_settings: dict) -> None:
        self.frames.reset()
        self.outbox.reset()
        self.capture = CaptureWorker(self.frames, source_settings)
        self.infer = InferenceWorker(self.frames, self.outbox, self.config)
        self.capture.status.connect(self.status)
        self.infer.status.connect(self.status)
        self.capture.finished.connect(self._on_finished)
        self.infer.finished.connect(self._on_finished)
        self.capture.start()
        self.infer.start()

    def request_stop(self) -> None:
        if self.capture is not None:
            self.capture.request_stop()
        if self.infer is not None:
            self.infer.request_stop()

    def set_paused(self, paused: bool) -> None:
        if self.capture is not None:
            self.capture.set_paused(paused)

    def seek(self, ratio: float) -> None:
        if self.capture is not None:
            self.capture.request_seek(ratio)

    def playback(self) -> dict:
        if self.capture is None:
            return {"time": 0.0, "duration": 0.0, "fps": 0.0, "capture_fps": 0.0}
        return self.capture.playback()

    def take_result(self):
        return self.outbox.pull()

    def _on_finished(self) -> None:
        if self.is_running():
            return
        if self._pending is not None:
            settings = self._pending
            self._pending = None
            self._launch(settings)
            return
        self.status.emit("Stopped")
        self.idle.emit()
