"""Detection, tracking, color, and overlay for one frame."""

from __future__ import annotations

import time

import cv2
import numpy as np

from color.color_analyzer import ColorAnalyzer
from inference.mmdet_backend import MMDetBackend, ModelNotReady
from inference.model_registry import ModelRegistry
from tracking.object_tracker import ObjectTracker
from utils.logger import get_logger
from visualization.renderer import render

log = get_logger("pipeline")


def _empty_color(obj: dict) -> None:
    obj["color_name"] = ""
    obj["rgb"] = None
    obj["hsv"] = None
    obj["lab"] = None
    obj["palette"] = []


class InferenceManager:
    def __init__(self, registry: ModelRegistry | None = None):
        self.registry = registry or ModelRegistry()
        self.backend = MMDetBackend()
        self.tracker = ObjectTracker()
        self.analyzer = ColorAnalyzer()
        self._model_for_tracks = ""
        self._color_tick = 0
        self._loose_palette: dict = {}
        self.on_status = None

    def shutdown(self) -> None:
        self.backend.unload()
        self.tracker.reset()

    def analyze(self, frame: np.ndarray, meta: dict, cfg: dict) -> dict:
        started = time.perf_counter()
        clean = np.ascontiguousarray(frame)
        timings = {
            "preprocess_ms": 0.0,
            "model_ms": 0.0,
            "tracker_ms": 0.0,
            "color_ms": 0.0,
            "render_ms": 0.0,
            "total_ms": 0.0,
        }
        objects: list[dict] = []
        error = ""
        model_name = self.backend.spec.display_name if self.backend.spec else ""
        device = cfg.get("device") or "cpu"
        need_model = bool(cfg.get("detection") or cfg.get("segmentation"))
        try:
            if need_model:
                objects, model_name, timings = self._detect(clean, cfg, timings)
            else:
                if self.backend.inferencer is not None:
                    self.backend.unload()
                self.tracker.reset()
        except Exception as exc:
            log.exception("Inference failed")
            error = str(exc)
            objects = []
        color_started = time.perf_counter()
        if cfg.get("color") and objects:
            self._apply_color(clean, objects, cfg)
        timings["color_ms"] = (time.perf_counter() - color_started) * 1000.0
        rendered, render_ms, public_objects = self._draw(clean, objects, cfg)
        timings["render_ms"] = render_ms
        timings["total_ms"] = (time.perf_counter() - started) * 1000.0
        classes = list(self.backend.class_names) or []
        return {
            "clean": clean,
            "rendered": rendered,
            "objects": public_objects,
            "timings": timings,
            "class_names": classes,
            "model_name": model_name,
            "device": self.backend.device if self.backend.inferencer is not None else device,
            "error": error,
            "meta": meta or {},
            "from_cache": False,
        }

    def render_cached(self, cache: dict, cfg: dict) -> dict:
        rendered, render_ms, _public = self._draw(cache["clean"], cache["objects"], cfg)
        output = dict(cache)
        output["rendered"] = rendered
        output["from_cache"] = True
        timings = dict(cache.get("timings") or {})
        timings["render_ms"] = render_ms
        output["timings"] = timings
        return output

    def _detect(self, clean: np.ndarray, cfg: dict, timings: dict):
        spec = self.registry.get(cfg.get("model_id") or "")
        if spec is None:
            raise ModelNotReady("No model is selected.")
        if not self.registry.is_installed(spec):
            raise ModelNotReady(f"{spec.display_name} is not installed. Open Model Manager and download it.")
        device = cfg.get("device") or "cpu"
        if self.backend.loaded_id != spec.model_id or self.backend.device != device:
            if self.on_status:
                self.on_status(f"Loading {spec.display_name} on {device}...")
            self.backend.load(spec, device)
            self.tracker.reset()
            self._model_for_tracks = spec.model_id
            if self.on_status:
                self.on_status(f"Model ready: {spec.display_name}")
        scale = float(cfg.get("processing_scale") or 1.0)
        scale = min(1.0, max(0.25, scale))
        prep = time.perf_counter()
        height, width = clean.shape[:2]
        if scale < 0.99:
            small_w = max(32, int(width * scale))
            small_h = max(32, int(height * scale))
            small = cv2.resize(clean, (small_w, small_h), interpolation=cv2.INTER_AREA)
        else:
            small = clean
            small_w, small_h = width, height
        timings["preprocess_ms"] = (time.perf_counter() - prep) * 1000.0
        user_thr = float(cfg.get("confidence") or 0.4)
        model_started = time.perf_counter()
        detections = self.backend.infer(small, user_thr)
        timings["model_ms"] = (time.perf_counter() - model_started) * 1000.0
        inv_x = width / float(small_w)
        inv_y = height / float(small_h)
        for det in detections:
            box = det.bbox.astype(np.float32)
            box[0] *= inv_x
            box[2] *= inv_x
            box[1] *= inv_y
            box[3] *= inv_y
            det.bbox = box
            if det.mask is not None:
                det.mask = cv2.resize(
                    np.asarray(det.mask).astype(np.uint8),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
        track_started = time.perf_counter()
        objects = self._associate(detections, cfg, user_thr, spec.model_id)
        timings["tracker_ms"] = (time.perf_counter() - track_started) * 1000.0
        return objects, spec.display_name, timings

    def _associate(self, detections, cfg: dict, user_thr: float, model_id: str):
        if model_id != self._model_for_tracks:
            self.tracker.reset()
            self._model_for_tracks = model_id
        if not cfg.get("tracking"):
            self.tracker.reset()
            objects = []
            visible = [det for det in detections if float(det.score) >= user_thr]
            for index, det in enumerate(visible, start=1):
                objects.append(self._from_detection(det, index, tracked=False))
            return objects
        self.tracker.configure(
            int(cfg.get("lost_frames_timeout") or 20),
            int(cfg.get("trail_length") or 30),
        )
        visible_tracks = self.tracker.update(detections, high_thr=user_thr, low_thr=0.10)
        return [self._from_track(track) for track in visible_tracks]

    def _from_detection(self, det, index: int, tracked: bool) -> dict:
        x1, y1, x2, y2 = [float(v) for v in det.bbox]
        return {
            "track_id": None,
            "select_key": f"det-{index}-{det.class_name}",
            "class_name": det.class_name,
            "score": float(det.score),
            "bbox": [x1, y1, x2, y2],
            "mask": det.mask,
            "color_name": "",
            "rgb": None,
            "hsv": None,
            "lab": None,
            "palette": [],
            "age": 1,
            "tracked": tracked,
            "motion": "",
            "velocity": [0.0, 0.0],
            "trail": [],
        }

    def _from_track(self, track) -> dict:
        x1, y1, x2, y2 = [float(v) for v in track.bbox]
        return {
            "track_id": int(track.track_id),
            "select_key": int(track.track_id),
            "class_name": track.class_name,
            "score": float(track.score),
            "bbox": [x1, y1, x2, y2],
            "mask": track.mask,
            "color_name": track.color_name,
            "rgb": track.rgb,
            "hsv": track.hsv,
            "lab": track.lab,
            "palette": list(track.palette),
            "age": int(track.age),
            "tracked": True,
            "motion": track.motion,
            "velocity": [float(track.velocity_x), float(track.velocity_y)],
            "trail": track.trail.as_list(),
            "_track": track,
        }

    def _apply_color(self, frame: np.ndarray, objects: list[dict], cfg: dict) -> None:
        interval = max(1, int(cfg.get("color_interval") or 5))
        want_palette = bool(cfg.get("palette"))
        use_mask = bool(cfg.get("segmentation"))
        self._color_tick += 1
        for obj in objects:
            track = obj.get("_track")
            refresh_palette = False
            cache_key = None
            if want_palette:
                if track is not None:
                    refresh_palette = track.palette_age >= interval or not track.palette
                else:
                    x1, y1, x2, y2 = obj["bbox"]
                    cache_key = (obj["class_name"], int((x1 + x2) / 40.0), int((y1 + y2) / 40.0))
                    refresh_palette = cache_key not in self._loose_palette or self._color_tick % interval == 0
            measured = self.analyzer.measure(
                frame,
                obj["bbox"],
                obj.get("mask") if use_mask else None,
                palette=bool(want_palette and refresh_palette),
            )
            if measured is None:
                continue
            palette = measured["palette"]
            if want_palette and not refresh_palette:
                if track is not None and track.palette:
                    palette = track.palette
                elif cache_key is not None and cache_key in self._loose_palette:
                    palette = self._loose_palette[cache_key]
            if track is not None:
                track.color_name = measured["name"]
                track.rgb = measured["rgb"]
                track.hsv = measured["hsv"]
                track.lab = measured["lab"]
                if refresh_palette:
                    track.palette = palette
                    track.palette_age = 0
                else:
                    track.palette_age += 1
            elif want_palette and refresh_palette and cache_key is not None:
                self._loose_palette[cache_key] = palette
            obj["color_name"] = measured["name"]
            obj["rgb"] = measured["rgb"]
            obj["hsv"] = measured["hsv"]
            obj["lab"] = measured["lab"]
            obj["palette"] = palette if want_palette else []
        if len(self._loose_palette) > 300:
            self._loose_palette.clear()

    def _draw(self, clean, objects, cfg):
        public = []
        for obj in objects:
            item = {key: value for key, value in obj.items() if key != "_track"}
            public.append(item)
        started = time.perf_counter()
        rendered = render(clean, public, cfg)
        return rendered, (time.perf_counter() - started) * 1000.0, public
