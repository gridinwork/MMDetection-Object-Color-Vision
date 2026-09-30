"""Main window of MMDetection Object & Color Vision Studio."""

from __future__ import annotations

import time

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.control_panel import ControlPanel
from app.model_manager_dialog import ModelManagerDialog
from app.pipeline import CameraScanThread, FpsMeter, VisionPipeline
from app.settings_dialog import SettingsDialog
from app.video_widget import VideoWidget
from inference.mode_manager import ModeManager
from inference.model_registry import COCO_CLASSES, ModelRegistry
from utils.gpu_info import device_status, format_gb, resolve_device
from utils.logger import get_logger
from utils.paths import objects_dir, screenshots_dir
from utils.settings_store import load_settings, save_settings

log = get_logger("gui")

IMAGE_FILTER = "Images (*.jpg *.jpeg *.png *.webp)"
VIDEO_FILTER = "Video (*.mp4 *.avi *.mov *.mkv)"


def _write_image(path, image: np.ndarray) -> None:
    ext = path.suffix.lower() or ".jpg"
    params = []
    if ext in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
    ok, encoded = cv2.imencode(ext, image, params)
    if not ok:
        raise RuntimeError(f"Could not encode {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def _stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _scale_token(value) -> str:
    number = float(value)
    if number >= 0.9:
        return "1.0"
    if number >= 0.6:
        return "0.75"
    return "0.5"


def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MMDetection Object & Color Vision Studio")
        self.resize(1480, 920)
        self.registry = ModelRegistry()
        self.mode_manager = ModeManager()
        self.pipeline = VisionPipeline()
        self._settings = load_settings()
        self._loading = True
        self._running = False
        self._paused = False
        self._task = "detection"
        self._color_generation = 0
        self._prev_modes = {}
        self._logged_modes = None
        self._logged_model = ""
        self._selected_key = None
        self._locked_id = None
        self._locked_caption = ""
        self._last = None
        self._last_error = ""
        self._warned_model = ""
        self._video_path = self._settings.get("last_video") or ""
        self._image_path = self._settings.get("last_image") or ""
        self._slider_held = False
        self._settings_dirty = False
        self._display_fps = FpsMeter()
        self._scan = None
        self._build()
        self._apply_settings()
        self._connect()
        self._loading = False
        self._push_config(interactive=False)
        self._scan = CameraScanThread()
        self._scan.found.connect(self._on_cameras)
        self._scan.start()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer)
        self._timer.start(33)
        QTimer.singleShot(0, self._populate_devices)
        geometry = self._settings.get("geometry")
        if isinstance(geometry, list) and len(geometry) == 4:
            self.setGeometry(*[int(v) for v in geometry])

    def _build(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        root.addWidget(self._build_toolbar())
        body = QHBoxLayout()
        self.video = VideoWidget()
        body.addWidget(self.video, 1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(430)
        self.panel = ControlPanel()
        scroll.setWidget(self.panel)
        body.addWidget(scroll)
        root.addLayout(body, 1)
        root.addWidget(self._build_video_bar())
        self.fps_label = QLabel("CAM —   INF —   DSP —   DROP —")
        self.statusBar().addPermanentWidget(self.fps_label)
        self.statusBar().showMessage("Ready")

    def _build_toolbar(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("toolbar")
        layout = QVBoxLayout(frame)
        row1 = QHBoxLayout()
        row2 = QHBoxLayout()
        self.source_combo = self._combo(row1, "SOURCE", [("Webcam", "webcam"), ("Video File", "video"), ("Image", "image")])
        self.camera_combo = self._combo(row1, "CAMERA", [("Camera 0", 0)])
        self.resolution_combo = self._combo(
            row1, "RESOLUTION", [("640x480", "640x480"), ("1280x720", "1280x720"), ("1920x1080", "1920x1080")]
        )
        self.scale_combo = self._combo(row1, "SCALE", [("100%", "1.0"), ("75%", "0.75"), ("50%", "0.5")])
        self.device_combo = self._combo(row1, "DEVICE", [("AUTO", "auto"), ("CPU", "cpu")])
        self.profile_combo = self._combo(
            row2, "MODEL", [("Auto", "auto"), ("Fast", "fast"), ("Balanced", "balanced"), ("Accurate", "accurate")]
        )
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(220)
        row2.addWidget(self.model_combo)
        self.optimization_combo = self._combo(
            row2,
            "OPTIMIZATION",
            [("MAX FPS", "max_fps"), ("BALANCED", "balanced"), ("MAX QUALITY", "max_quality")],
        )
        self.browse_button = QPushButton("BROWSE")
        self.path_label = QLabel("No file")
        self.path_label.setObjectName("pathLabel")
        row2.addWidget(self.browse_button)
        row2.addWidget(self.path_label, 1)
        self.start_button = QPushButton("START")
        self.stop_button = QPushButton("STOP")
        self.pause_button = QPushButton("PAUSE")
        self.shot_button = QPushButton("SAVE SCREENSHOT")
        self.clean_button = QPushButton("SAVE CLEAN FRAME")
        self.models_button = QPushButton("MODEL MANAGER")
        self.settings_button = QPushButton("SETTINGS")
        self.start_button.setObjectName("startButton")
        self.stop_button.setObjectName("stopButton")
        for button in (
            self.start_button,
            self.stop_button,
            self.pause_button,
            self.shot_button,
            self.clean_button,
            self.models_button,
            self.settings_button,
        ):
            row2.addWidget(button)
        layout.addLayout(row1)
        layout.addLayout(row2)
        self.stop_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        return frame

    def _combo(self, row, title, items) -> QComboBox:
        row.addWidget(QLabel(title))
        combo = QComboBox()
        for label, value in items:
            combo.addItem(label, value)
        row.addWidget(combo)
        return combo

    def _build_video_bar(self) -> QFrame:
        self.video_bar = QFrame()
        self.video_bar.setObjectName("videoBar")
        layout = QHBoxLayout(self.video_bar)
        self.time_label = QLabel("00:00 / 00:00")
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 1000)
        layout.addWidget(self.time_label)
        layout.addWidget(self.timeline, 1)
        self.video_bar.setVisible(False)
        return self.video_bar

    def _connect(self) -> None:
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        self.camera_combo.currentIndexChanged.connect(self._on_source_field)
        self.resolution_combo.currentIndexChanged.connect(self._on_source_field)
        self.scale_combo.currentIndexChanged.connect(self._on_runtime_changed)
        self.device_combo.currentIndexChanged.connect(self._on_runtime_changed)
        self.profile_combo.currentIndexChanged.connect(self._on_runtime_changed)
        self.model_combo.currentIndexChanged.connect(self._on_runtime_changed)
        self.optimization_combo.currentIndexChanged.connect(self._on_optimization)
        self.browse_button.clicked.connect(self._browse)
        self.start_button.clicked.connect(self._on_start)
        self.stop_button.clicked.connect(self._on_stop)
        self.pause_button.clicked.connect(self._on_pause)
        self.shot_button.clicked.connect(self._save_screenshot)
        self.clean_button.clicked.connect(self._save_clean)
        self.models_button.clicked.connect(self._open_models)
        self.settings_button.clicked.connect(self._open_settings)
        self.panel.settings_changed.connect(self._on_runtime_changed)
        self.panel.lock_requested.connect(self._on_lock)
        self.panel.save_object_requested.connect(lambda: self._save_object(False))
        self.panel.save_masked_requested.connect(lambda: self._save_object(True))
        self.panel.object_selected.connect(self._select_key)
        self.video.clicked.connect(self._on_video_click)
        self.timeline.sliderPressed.connect(lambda: setattr(self, "_slider_held", True))
        self.timeline.sliderReleased.connect(self._on_seek)
        self.pipeline.status.connect(self._on_status)
        self.pipeline.idle.connect(self._on_idle)

    def _apply_settings(self) -> None:
        settings = self._settings
        self._set_combo(self.source_combo, settings.get("source", "webcam"))
        self._set_combo(self.resolution_combo, settings.get("resolution", "1280x720"))
        self._set_combo(self.scale_combo, _scale_token(settings.get("processing_scale") or 0.75))
        self._set_combo(self.profile_combo, settings.get("profile", "fast"))
        self._set_combo(self.optimization_combo, settings.get("optimization", "balanced"))
        self.panel.set_modes(settings.get("modes") or {})
        self.panel.set_confidence(float(settings.get("confidence") or 0.4))
        self.panel.set_class_names(list(COCO_CLASSES), list(settings.get("disabled_classes") or []))
        self._task = "instance_segmentation" if self.panel.modes().get("segmentation") else "detection"
        self._refresh_model_combo()
        explicit = settings.get("explicit_model_id") or ""
        if explicit:
            index = self.model_combo.findData(explicit)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
        self._update_source_widgets()

    def _set_combo(self, combo: QComboBox, value) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _populate_devices(self) -> None:
        from utils.gpu_info import list_devices

        current = self._settings.get("device", "auto")
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        self.device_combo.addItem("AUTO", "auto")
        for device in list_devices():
            self.device_combo.addItem(device["label"], device["id"])
        index = self.device_combo.findData(current)
        self.device_combo.setCurrentIndex(index if index >= 0 else 0)
        self.device_combo.blockSignals(False)
        self._push_config(interactive=False)

    def _on_cameras(self, cameras) -> None:
        saved = int(self._settings.get("camera_index") or 0)
        indexes = [int(index) for index in cameras] or [0]
        if saved not in indexes:
            indexes.append(saved)
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        for index in indexes:
            self.camera_combo.addItem(f"Camera {index}", index)
        current = self.camera_combo.findData(saved)
        self.camera_combo.setCurrentIndex(current if current >= 0 else 0)
        self.camera_combo.blockSignals(False)

    def _refresh_model_combo(self) -> None:
        current = self.model_combo.currentData()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItem("Follow profile", "")
        for spec in self.registry.by_task(self._task):
            label = spec.display_name
            if not self.registry.is_installed(spec):
                label += " — not installed"
            self.model_combo.addItem(label, spec.model_id)
        index = self.model_combo.findData(current if current else "")
        self.model_combo.setCurrentIndex(index if index >= 0 else 0)
        self.model_combo.blockSignals(False)

    def _on_source_changed(self) -> None:
        self._update_source_widgets()
        self._on_source_field()

    def _update_source_widgets(self) -> None:
        kind = self.source_combo.currentData()
        self.camera_combo.setEnabled(kind == "webcam")
        self.resolution_combo.setEnabled(kind == "webcam")
        self.browse_button.setVisible(kind in {"video", "image"})
        self.path_label.setVisible(kind in {"video", "image"})
        self.video_bar.setVisible(kind == "video")
        path = self._video_path if kind == "video" else self._image_path
        self.path_label.setText(path or "No file")

    def _on_source_field(self) -> None:
        self._settings_dirty = True
        if self._running:
            self.statusBar().showMessage("Press START to apply the new source.")

    def _on_optimization(self) -> None:
        scale = {"max_fps": "0.5", "balanced": "0.75", "max_quality": "1.0"}.get(
            self.optimization_combo.currentData(), "0.75"
        )
        self._set_combo(self.scale_combo, scale)
        self._on_runtime_changed()

    def _on_runtime_changed(self) -> None:
        if self._loading:
            return
        self._settings_dirty = True
        self._push_config(interactive=True)

    def _push_config(self, interactive: bool) -> None:
        modes = self.panel.modes()
        if modes != self._prev_modes:
            if self._prev_modes:
                if modes.get("color") != self._prev_modes.get("color") or modes.get("palette") != self._prev_modes.get("palette"):
                    self._color_generation += 1
            self._prev_modes = dict(modes)
            if modes != self._logged_modes:
                log.info("Mode change %s", modes)
                self._logged_modes = dict(modes)
        task = "instance_segmentation" if modes.get("segmentation") else "detection"
        if task != self._task:
            self._task = task
            self._refresh_model_combo()
        explicit = self.model_combo.currentData() or ""
        try:
            spec = self.mode_manager.resolve(
                self.registry,
                segmentation=bool(modes.get("segmentation")),
                profile=str(self.profile_combo.currentData() or "fast"),
                optimization=str(self.optimization_combo.currentData() or "balanced"),
                explicit_model_id=str(explicit),
            )
        except LookupError as exc:
            self.statusBar().showMessage(str(exc))
            return
        if spec.model_id != self._logged_model:
            log.info("Selected model %s", spec.model_id)
            self._logged_model = spec.model_id
        if not self.registry.is_installed(spec):
            message = f"{spec.display_name} is not installed. Open Model Manager."
            self.statusBar().showMessage(message)
            if interactive and self._warned_model != spec.model_id:
                self._warned_model = spec.model_id
                QMessageBox.information(self, "Model", message)
        else:
            self._warned_model = ""
        device_choice = str(self.device_combo.currentData() or "auto")
        cfg = {
            "model_id": spec.model_id,
            "device": resolve_device(device_choice),
            "confidence": self.panel.confidence(),
            "detection": bool(modes.get("detection")),
            "segmentation": bool(modes.get("segmentation")),
            "tracking": bool(modes.get("tracking")),
            "color": bool(modes.get("color")),
            "trails": bool(modes.get("trails")),
            "labels": bool(modes.get("labels")),
            "show_confidence": bool(modes.get("confidence")),
            "show_id": bool(modes.get("object_id")),
            "detailed_color": bool(modes.get("detailed_color")),
            "palette": bool(modes.get("palette")),
            "processing_scale": float(self.scale_combo.currentData() or 1.0),
            "disabled_classes": self.panel.disabled_classes(),
            "lost_frames_timeout": int(self._settings.get("lost_frames_timeout") or 20),
            "color_interval": int(self._settings.get("color_interval") or 5),
            "trail_length": int(self._settings.get("trail_length") or 30),
            "locked_id": self._locked_id,
            "selected_id": self._selected_key,
            "locked_caption": self._locked_caption,
            "color_generation": self._color_generation,
        }
        self.pipeline.config.replace(cfg)
        self.panel.debug_box.setVisible(bool(modes.get("debug")))

    def _on_start(self) -> None:
        kind = self.source_combo.currentData()
        if kind == "video" and not self._video_path:
            self._browse()
            if not self._video_path:
                return
        if kind == "image" and not self._image_path:
            self._browse()
            if not self._image_path:
                return
        width, height = self._resolution()
        settings = {"kind": kind, "index": int(self.camera_combo.currentData() or 0), "width": width, "height": height}
        if kind == "video":
            settings["path"] = self._video_path
        elif kind == "image":
            settings["path"] = self._image_path
        self._push_config(interactive=True)
        self._paused = False
        self.pause_button.setText("PAUSE")
        self.pipeline.start(settings)
        self._running = True
        self._update_buttons()
        log.info("Source start %s", settings)
        if kind == "webcam":
            self.video.set_message(
                f"OPENING CAMERA {settings['index']}\n"
                f"{width}x{height}\n\n"
                "The camera is switching mode.\n"
                "This can take a few seconds.\n"
                "The program is not frozen."
            )
            self.statusBar().showMessage(f"Opening camera {settings['index']} at {width}x{height}...")
        else:
            self.video.set_message("LOADING...")
            self.statusBar().showMessage("Starting...")

    def _on_stop(self) -> None:
        self._paused = False
        self.pause_button.setText("PAUSE")
        self.pipeline.request_stop()
        self.statusBar().showMessage("Stopping...")
        log.info("Source stop")

    def _on_pause(self) -> None:
        self._paused = not self._paused
        self.pipeline.set_paused(self._paused)
        self.pause_button.setText("RESUME" if self._paused else "PAUSE")

    def _on_idle(self) -> None:
        self._running = False
        self._paused = False
        self.pause_button.setText("PAUSE")
        self._update_buttons()

    def _update_buttons(self) -> None:
        self.start_button.setEnabled(not self._running)
        self.stop_button.setEnabled(self._running)
        self.pause_button.setEnabled(self._running)

    def _on_status(self, text: str) -> None:
        self.statusBar().showMessage(text)
        if text == "Video ended":
            self._paused = True
            self.pause_button.setText("RESUME")

    def _browse(self) -> None:
        kind = self.source_combo.currentData()
        if kind == "video":
            path, _ = QFileDialog.getOpenFileName(self, "Open video", self._video_path, VIDEO_FILTER)
            if path:
                self._video_path = path
                self.path_label.setText(path)
                self._settings_dirty = True
        elif kind == "image":
            path, _ = QFileDialog.getOpenFileName(self, "Open image", self._image_path, IMAGE_FILTER)
            if path:
                self._image_path = path
                self.path_label.setText(path)
                self._settings_dirty = True
                if self._running:
                    self._on_start()

    def _on_timer(self) -> None:
        result = self.pipeline.take_result()
        if result is not None:
            self._apply_result(result)
        if self.source_combo.currentData() == "video" and not self._slider_held:
            play = self.pipeline.playback()
            duration = float(play.get("duration") or 0.0)
            current = float(play.get("time") or 0.0)
            self.time_label.setText(f"{_clock(current)} / {_clock(duration)}")
            if duration > 0:
                self.timeline.setValue(int(max(0.0, min(1.0, current / duration)) * 1000))
        if self._settings_dirty and self._timer.interval() > 0:
            self._save_tick = getattr(self, "_save_tick", 0) + 1
            if self._save_tick % 90 == 0:
                self._save_settings()

    def _apply_result(self, result: dict) -> None:
        self._last = result
        rendered = result.get("rendered")
        if rendered is not None:
            self.video.set_frame(rendered)
            self._display_fps.tick()
        names = result.get("class_names") or []
        if names:
            self.panel.set_class_names(list(names), self.panel.disabled_classes())
        objects = result.get("objects") or []
        detected = sorted({obj["class_name"] for obj in objects})
        self.panel.set_detected(detected)
        self.panel.set_objects(objects, self._selected_key)
        selected = self._find(self._selected_key)
        self.panel.show_object(selected, self._is_locked(selected))
        error = result.get("error") or ""
        if error and error != self._last_error:
            self.statusBar().showMessage(error)
        self._last_error = error
        self._update_hud(result)

    def _update_hud(self, result: dict) -> None:
        meta = result.get("meta") or {}
        timings = result.get("timings") or {}
        device = device_status(result.get("device") or "")
        capture_fps = float(meta.get("capture_fps") or self.pipeline.playback().get("capture_fps") or 0.0)
        infer_fps = float(result.get("inference_fps") or 0.0)
        display_fps = self._display_fps.value
        dropped = int(meta.get("dropped") or 0)
        self.fps_label.setText(
            f"CAM {capture_fps:4.1f}   INF {infer_fps:4.1f}   DSP {display_fps:4.1f}   DROP {dropped}"
        )
        vram = "—"
        if device["total"]:
            vram = f"{format_gb(device['used'])} / {format_gb(device['total'])}"
        self.panel.set_gpu(f"GPU NAME: {device['name']}\nVRAM USED / TOTAL: {vram}\nDEVICE: {device['device']}")
        if self.panel.modes().get("debug"):
            lines = [
                f"FRAME {meta.get('frame_index', 0)}",
                f"CAPTURE FPS {capture_fps:.1f}",
                f"INFERENCE FPS {infer_fps:.1f}",
                f"DISPLAY FPS {display_fps:.1f}",
                f"PREPROCESS MS {timings.get('preprocess_ms', 0):.1f}",
                f"MODEL MS {timings.get('model_ms', 0):.1f}",
                f"TRACKER MS {timings.get('tracker_ms', 0):.1f}",
                f"COLOR MS {timings.get('color_ms', 0):.1f}",
                f"RENDER MS {timings.get('render_ms', 0):.1f}",
                f"TOTAL MS {timings.get('total_ms', 0):.1f}",
                f"QUEUE SIZE {meta.get('queue_size', 0)}",
                f"DROPPED FRAMES {dropped}",
                f"GPU {device['name']}",
                f"VRAM {vram}",
                f"ACTIVE MODEL {result.get('model_name') or '—'}",
            ]
            selected = self._find(self._selected_key)
            if selected and selected.get("motion"):
                vx, vy = selected.get("velocity") or (0.0, 0.0)
                lines.append(f"MOTION {selected['motion']}")
                lines.append(f"VELOCITY {vx:.2f} {vy:.2f}")
            self.panel.set_debug("\n".join(lines), True)

    def _on_video_click(self, x: float, y: float) -> None:
        obj = self._hit(x, y)
        self._select_key(None if obj is None else obj.get("select_key"))

    def _select_key(self, key) -> None:
        self._selected_key = key
        selected = self._find(key)
        self.panel.show_object(selected, self._is_locked(selected))
        if self._last is not None:
            self.panel.set_objects(self._last.get("objects") or [], key)
        self._push_config(interactive=False)

    def _on_lock(self) -> None:
        selected = self._find(self._selected_key)
        if selected is None or not selected.get("tracked") or selected.get("track_id") is None:
            self.statusBar().showMessage("Turn on Object Tracking and select a tracked object.")
            return
        track_id = int(selected["track_id"])
        if self._locked_id == track_id:
            self._locked_id = None
            self._locked_caption = ""
        else:
            self._locked_id = track_id
            self._locked_caption = f"TARGET: {str(selected['class_name']).upper()} #{track_id}"
        self.panel.show_object(selected, self._is_locked(selected))
        self._push_config(interactive=False)

    def _is_locked(self, obj) -> bool:
        return bool(obj and obj.get("track_id") is not None and self._locked_id == int(obj["track_id"]))

    def _find(self, key):
        if key is None or self._last is None:
            return None
        for obj in self._last.get("objects") or []:
            if obj.get("select_key") == key:
                return obj
        return None

    def _hit(self, x: float, y: float):
        if self._last is None:
            return None
        disabled = set(self.panel.disabled_classes())
        hits = []
        for obj in self._last.get("objects") or []:
            if obj.get("class_name") in disabled:
                continue
            x1, y1, x2, y2 = [float(v) for v in obj["bbox"]]
            if x1 <= x <= x2 and y1 <= y <= y2:
                hits.append(((x2 - x1) * (y2 - y1), obj))
        if not hits:
            return None
        hits.sort(key=lambda item: item[0])
        return hits[0][1]

    def _on_seek(self) -> None:
        self._slider_held = False
        self.pipeline.seek(self.timeline.value() / 1000.0)

    def _save_screenshot(self) -> None:
        if not self._last or self._last.get("rendered") is None:
            self.statusBar().showMessage("No frame yet.")
            return
        path = screenshots_dir() / f"mmdet_{_stamp()}.jpg"
        _write_image(path, self._last["rendered"])
        self.statusBar().showMessage(f"Saved {path}")
        log.info("Screenshot %s", path)

    def _save_clean(self) -> None:
        if not self._last or self._last.get("clean") is None:
            self.statusBar().showMessage("No frame yet.")
            return
        path = screenshots_dir() / f"mmdet_clean_{_stamp()}.jpg"
        _write_image(path, self._last["clean"])
        self.statusBar().showMessage(f"Saved {path}")
        log.info("Clean frame %s", path)

    def _save_object(self, masked: bool) -> None:
        selected = self._find(self._selected_key)
        if selected is None or self._last is None or self._last.get("clean") is None:
            self.statusBar().showMessage("Select an object first.")
            return
        image = self._last["clean"]
        height, width = image.shape[:2]
        x1, y1, x2, y2 = [int(round(float(v))) for v in selected["bbox"]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            self.statusBar().showMessage("Object crop is empty.")
            return
        crop = image[y1:y2, x1:x2].copy()
        class_name = str(selected.get("class_name") or "object").replace(" ", "_")
        track = selected.get("track_id")
        ident = f"ID{track}" if track is not None else "frame"
        if masked:
            mask = selected.get("mask")
            if mask is None:
                self.statusBar().showMessage("This object has no segmentation mask.")
                return
            mask_arr = np.asarray(mask)
            if mask_arr.shape[:2] != image.shape[:2]:
                mask_arr = cv2.resize(mask_arr.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
            alpha = (mask_arr[y1:y2, x1:x2] > 0).astype(np.uint8) * 255
            bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = alpha
            path = objects_dir() / f"{class_name}_{ident}_masked_{_stamp()}.png"
            _write_image(path, bgra)
        else:
            path = objects_dir() / f"{class_name}_{ident}_{_stamp()}.png"
            _write_image(path, crop)
        self.statusBar().showMessage(f"Saved {path}")
        log.info("Object crop %s", path)

    def _open_models(self) -> None:
        dialog = ModelManagerDialog(self.registry, self)
        dialog.changed.connect(self._models_changed)
        dialog.exec()
        self._models_changed()

    def _models_changed(self) -> None:
        self.registry.reload()
        self._refresh_model_combo()
        self._push_config(interactive=False)

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self._settings, self)
        if dialog.exec() != SettingsDialog.DialogCode.Accepted:
            return
        self._settings.update(dialog.values())
        self._settings_dirty = True
        self._color_generation += 1
        self._push_config(interactive=False)
        self._save_settings()

    def _resolution(self) -> tuple[int, int]:
        text = str(self.resolution_combo.currentData() or "1280x720")
        width, height = text.lower().split("x")
        return int(width), int(height)

    def _collect_settings(self) -> dict:
        rect = self.geometry()
        modes = self.panel.modes()
        return {
            "source": self.source_combo.currentData(),
            "camera_index": int(self.camera_combo.currentData() or 0),
            "resolution": self.resolution_combo.currentData(),
            "device": self.device_combo.currentData() or "auto",
            "profile": self.profile_combo.currentData(),
            "explicit_model_id": self.model_combo.currentData() or "",
            "optimization": self.optimization_combo.currentData(),
            "confidence": self.panel.confidence(),
            "processing_scale": float(self.scale_combo.currentData() or 1.0),
            "last_video": self._video_path,
            "last_image": self._image_path,
            "modes": modes,
            "disabled_classes": self.panel.disabled_classes(),
            "lost_frames_timeout": int(self._settings.get("lost_frames_timeout") or 20),
            "color_interval": int(self._settings.get("color_interval") or 5),
            "trail_length": int(self._settings.get("trail_length") or 30),
            "geometry": [rect.x(), rect.y(), rect.width(), rect.height()],
        }

    def _save_settings(self) -> None:
        self._settings = self._collect_settings()
        save_settings(self._settings)
        self._settings_dirty = False

    def closeEvent(self, event) -> None:
        self._save_settings()
        log.info("Shutdown")
        self.pipeline.request_stop()
        deadline = time.time() + 20
        while self.pipeline.is_running() and time.time() < deadline:
            QApplication.processEvents()
            time.sleep(0.05)
        event.accept()


def run_app() -> int:
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    from app.style import STYLESHEET

    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec()
