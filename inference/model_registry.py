"""Known MMDetection models and optional custom entries."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from utils.logger import get_logger
from utils.paths import checkpoints_dir, custom_registry_file, project_root

log = get_logger("models")

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)


@dataclass
class ModelSpec:
    model_id: str
    display_name: str
    task: str
    config_name: str
    checkpoint_name: str
    weights_url: str
    input_size: int
    supports_masks: bool
    description: str
    profile: str | None = None
    custom: bool = False

    def checkpoint_path(self) -> Path:
        raw = Path(self.checkpoint_name)
        if raw.is_absolute():
            return raw
        if self.custom:
            return project_root() / raw
        return checkpoints_dir() / raw.name


def _builtin() -> list[ModelSpec]:
    return [
        ModelSpec(
            model_id="rtmdet_tiny_8xb32-300e_coco",
            display_name="RTMDet Tiny (Fast)",
            task="detection",
            config_name="rtmdet_tiny_8xb32-300e_coco",
            checkpoint_name="rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
            weights_url="https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
            input_size=640,
            supports_masks=False,
            description="Lightweight RTMDet for the highest frame rate. COCO box AP 40.9.",
            profile="fast",
        ),
        ModelSpec(
            model_id="rtmdet_s_8xb32-300e_coco",
            display_name="RTMDet S (Balanced)",
            task="detection",
            config_name="rtmdet_s_8xb32-300e_coco",
            checkpoint_name="rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth",
            weights_url="https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_s_8xb32-300e_coco/rtmdet_s_8xb32-300e_coco_20220905_161602-387a891e.pth",
            input_size=640,
            supports_masks=False,
            description="RTMDet-S, the speed/quality compromise. COCO box AP 44.5.",
            profile="balanced",
        ),
        ModelSpec(
            model_id="rtmdet_m_8xb32-300e_coco",
            display_name="RTMDet M",
            task="detection",
            config_name="rtmdet_m_8xb32-300e_coco",
            checkpoint_name="rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth",
            weights_url="https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth",
            input_size=640,
            supports_masks=False,
            description="RTMDet-M. COCO box AP 49.1.",
            profile=None,
        ),
        ModelSpec(
            model_id="rtmdet_l_8xb32-300e_coco",
            display_name="RTMDet L (Accurate)",
            task="detection",
            config_name="rtmdet_l_8xb32-300e_coco",
            checkpoint_name="rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth",
            weights_url="https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_l_8xb32-300e_coco/rtmdet_l_8xb32-300e_coco_20220719_112030-5a0be7c4.pth",
            input_size=640,
            supports_masks=False,
            description="RTMDet-L from the current MMDetection inference guide. COCO box AP 51.3.",
            profile="accurate",
        ),
        ModelSpec(
            model_id="rtmdet-ins_tiny_8xb32-300e_coco",
            display_name="RTMDet-Ins Tiny (Fast)",
            task="instance_segmentation",
            config_name="rtmdet-ins_tiny_8xb32-300e_coco",
            checkpoint_name="rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth",
            weights_url="https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth",
            input_size=640,
            supports_masks=True,
            description="Realtime instance segmentation. Boxes and masks from one model. Mask AP 35.4.",
            profile="fast",
        ),
        ModelSpec(
            model_id="rtmdet-ins_s_8xb32-300e_coco",
            display_name="RTMDet-Ins S (Balanced)",
            task="instance_segmentation",
            config_name="rtmdet-ins_s_8xb32-300e_coco",
            checkpoint_name="rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth",
            weights_url="https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_s_8xb32-300e_coco/rtmdet-ins_s_8xb32-300e_coco_20221121_212604-fdc5d7ec.pth",
            input_size=640,
            supports_masks=True,
            description="Balanced RTMDet instance segmentation. Mask AP 38.7.",
            profile="balanced",
        ),
        ModelSpec(
            model_id="rtmdet-ins_l_8xb32-300e_coco",
            display_name="RTMDet-Ins L (Accurate)",
            task="instance_segmentation",
            config_name="rtmdet-ins_l_8xb32-300e_coco",
            checkpoint_name="rtmdet-ins_l_8xb32-300e_coco_20221124_103237-78d1d652.pth",
            weights_url="https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_l_8xb32-300e_coco/rtmdet-ins_l_8xb32-300e_coco_20221124_103237-78d1d652.pth",
            input_size=640,
            supports_masks=True,
            description="Heavier RTMDet instance segmentation. Mask AP 43.7.",
            profile="accurate",
        ),
        ModelSpec(
            model_id="mask-rcnn_r50_fpn_1x_coco",
            display_name="Mask R-CNN R50",
            task="instance_segmentation",
            config_name="mask-rcnn_r50_fpn_1x_coco",
            checkpoint_name="mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth",
            weights_url="https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth",
            input_size=800,
            supports_masks=True,
            description="Classic Mask R-CNN ResNet-50. Accurate masks, slower than RTMDet-Ins.",
            profile=None,
        ),
    ]


class ModelRegistry:
    def __init__(self):
        self._models: dict[str, ModelSpec] = {spec.model_id: spec for spec in _builtin()}
        self._load_custom()

    def _load_custom(self) -> None:
        path = custom_registry_file()
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            log.exception("Could not read custom model registry")
            return
        if not isinstance(payload, list):
            log.error("custom_models/registry.json must be a list")
            return
        for item in payload:
            if not isinstance(item, dict) or not item.get("model_id"):
                continue
            spec = ModelSpec(
                model_id=str(item["model_id"]),
                display_name=str(item.get("display_name") or item["model_id"]),
                task=str(item.get("task") or "detection"),
                config_name=str(item.get("config_path") or item.get("config_name") or ""),
                checkpoint_name=str(item.get("checkpoint_path") or item.get("checkpoint_name") or ""),
                weights_url=str(item.get("weights_url") or ""),
                input_size=int(item.get("input_size") or 640),
                supports_masks=bool(item.get("supports_masks")),
                description=str(item.get("description") or "Custom model"),
                profile=item.get("profile") or None,
                custom=True,
            )
            self._models[spec.model_id] = spec
            log.info("Registered custom model %s", spec.model_id)

    def reload(self) -> None:
        self._models = {spec.model_id: spec for spec in _builtin()}
        self._load_custom()

    def all(self) -> list[ModelSpec]:
        return list(self._models.values())

    def get(self, model_id: str) -> ModelSpec | None:
        return self._models.get(model_id)

    def by_task(self, task: str) -> list[ModelSpec]:
        return [spec for spec in self._models.values() if spec.task == task]

    def for_profile(self, task: str, profile: str) -> ModelSpec | None:
        for spec in self._models.values():
            if spec.task == task and spec.profile == profile and not spec.custom:
                return spec
        return None

    def is_installed(self, spec: ModelSpec) -> bool:
        path = spec.checkpoint_path()
        return path.is_file() and path.stat().st_size > 1_000_000

    def status_text(self, spec: ModelSpec) -> str:
        if self.is_installed(spec):
            return "INSTALLED"
        return "NOT INSTALLED"

    def verify(self, spec: ModelSpec) -> tuple[bool, str]:
        path = spec.checkpoint_path()
        if not path.is_file():
            return False, "File is missing"
        size = path.stat().st_size
        if size < 1_000_000:
            return False, f"File is too small ({size} bytes)"
        with path.open("rb") as handle:
            magic = handle.read(4)
        if magic == b"PK\x03\x04":
            import zipfile

            try:
                with zipfile.ZipFile(path) as archive:
                    names = archive.namelist()
            except zipfile.BadZipFile:
                return False, "Corrupt zip archive"
            if not any("data.pkl" in name for name in names):
                return False, "Checkpoint archive has no data.pkl"
            return True, f"OK, {size / (1024 * 1024):.1f} MB"
        if magic[:1] == b"\x80":
            return True, f"OK, PyTorch pickle checkpoint, {size / (1024 * 1024):.1f} MB"
        return False, "Not a PyTorch checkpoint"

    def delete_checkpoint(self, spec: ModelSpec) -> tuple[bool, str]:
        path = spec.checkpoint_path()
        try:
            path.resolve().relative_to(project_root().resolve())
        except ValueError:
            return False, "Refusing to delete a file outside the project"
        if not path.is_file():
            return False, "File is already missing"
        path.unlink()
        log.info("Deleted checkpoint %s", path)
        return True, "Deleted"
