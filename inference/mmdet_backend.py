"""MMDetection DetInferencer wrapper.

The call shape follows the current MMDetection user guide
(docs/en/user_guides/inference.md) and mmdet/apis/det_inferencer.py:
DetInferencer(model, weights, device, show_progress=False), then
inferencer(image_bgr, pred_score_thr=..., return_datasamples=True).
pred_score_thr only affects MMDetection's own drawing, so scores are
filtered here as well.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np

from inference.model_registry import ModelSpec
from utils.logger import get_logger

log = get_logger("inference")


@dataclass
class Detection:
    bbox: np.ndarray
    score: float
    label: int
    class_name: str
    mask: np.ndarray | None = None


class ModelNotReady(RuntimeError):
    pass


def _as_numpy_masks(masks, count: int):
    if masks is None:
        return None
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and torch.is_tensor(masks):
        array = masks.detach().cpu().numpy()
    elif hasattr(masks, "to_ndarray"):
        array = masks.to_ndarray()
    else:
        array = np.asarray(masks)
    if array.ndim == 2 and count == 1:
        array = array[None, ...]
    return array


def _set_score_thr(cfg, thr: float, seen: set | None = None) -> None:
    if cfg is None:
        return
    if seen is None:
        seen = set()
    marker = id(cfg)
    if marker in seen:
        return
    seen.add(marker)
    try:
        keys = list(cfg.keys())
    except Exception:
        if hasattr(cfg, "score_thr"):
            try:
                cfg.score_thr = float(thr)
            except Exception:
                return
        return
    for key in keys:
        try:
            value = cfg[key]
        except Exception:
            continue
        if key in {"score_thr", "score_threshold"}:
            try:
                cfg[key] = float(thr)
            except Exception:
                try:
                    setattr(cfg, key, float(thr))
                except Exception:
                    continue
        elif hasattr(value, "keys") or hasattr(value, "score_thr"):
            _set_score_thr(value, thr, seen)


class MMDetBackend:
    def __init__(self):
        self.inferencer = None
        self.spec: ModelSpec | None = None
        self.device = "cpu"
        self.class_names: tuple[str, ...] = ()
        self._floor = 0.10

    @property
    def loaded_id(self) -> str:
        return "" if self.spec is None else self.spec.model_id

    def load(self, spec: ModelSpec, device: str) -> None:
        if (
            self.inferencer is not None
            and self.spec is not None
            and self.spec.model_id == spec.model_id
            and self.device == device
        ):
            return
        self.unload()
        weights = spec.checkpoint_path()
        if not weights.is_file():
            raise ModelNotReady(f"{spec.display_name} is not installed.")
        log.info("Loading model %s on %s", spec.model_id, device)
        from mmdet.apis import DetInferencer

        model_ref = spec.config_name or spec.model_id
        self.inferencer = DetInferencer(
            model=model_ref,
            weights=str(weights),
            device=device,
            show_progress=False,
        )
        self.spec = spec
        self.device = device
        meta = getattr(self.inferencer.model, "dataset_meta", {}) or {}
        classes = meta.get("classes") or ()
        self.class_names = tuple(str(name) for name in classes)
        self.set_score_floor(self._floor)
        if str(device).startswith("cuda"):
            import torch

            torch.backends.cudnn.benchmark = True
        log.info("Model loaded: %s classes=%s", spec.display_name, len(self.class_names))

    def unload(self) -> None:
        if self.inferencer is None and self.spec is None:
            return
        name = self.spec.display_name if self.spec else "model"
        self.inferencer = None
        self.spec = None
        self.class_names = ()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            log.exception("CUDA cache clear failed")
        log.info("Model unloaded: %s", name)

    def set_score_floor(self, floor: float) -> None:
        self._floor = float(floor)
        if self.inferencer is None:
            return
        _set_score_thr(getattr(self.inferencer.model, "test_cfg", None), self._floor)

    def infer(self, image_bgr: np.ndarray, score_thr: float) -> list[Detection]:
        if self.inferencer is None or self.spec is None:
            raise ModelNotReady("Model is not loaded.")
        image = np.ascontiguousarray(image_bgr)
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if image.ndim == 2:
            import cv2

            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        floor = min(float(score_thr), self._floor)
        self.set_score_floor(floor)
        import torch

        with torch.inference_mode():
            output = self.inferencer(
                image,
                batch_size=1,
                return_vis=False,
                show=False,
                wait_time=0,
                no_save_vis=True,
                draw_pred=False,
                pred_score_thr=float(score_thr),
                return_datasamples=True,
                print_result=False,
                no_save_pred=True,
                out_dir="",
            )
        predictions = output.get("predictions") or []
        if not predictions:
            return []
        sample = predictions[0]
        instances = sample.pred_instances
        bboxes = instances.bboxes
        scores = instances.scores
        labels = instances.labels
        if torch.is_tensor(bboxes):
            bboxes = bboxes.detach().cpu().numpy()
        if torch.is_tensor(scores):
            scores = scores.detach().cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()
        bboxes = np.asarray(bboxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)
        raw_masks = instances.get("masks") if hasattr(instances, "get") else getattr(instances, "masks", None)
        masks = _as_numpy_masks(raw_masks, len(bboxes)) if self.spec.supports_masks else None
        detections = []
        # Keep scores down to the tracker floor. Display filtering happens later.
        for index in range(len(bboxes)):
            score = float(scores[index])
            if score < floor:
                continue
            label = int(labels[index])
            if 0 <= label < len(self.class_names):
                class_name = self.class_names[label]
            else:
                class_name = str(label)
            mask = None
            if masks is not None and index < len(masks):
                mask = np.asarray(masks[index])
            detections.append(
                Detection(
                    bbox=bboxes[index].astype(np.float32),
                    score=score,
                    label=label,
                    class_name=class_name,
                    mask=mask,
                )
            )
        return detections
