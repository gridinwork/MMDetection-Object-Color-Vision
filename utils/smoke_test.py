"""Import check, CUDA check, one real detection, tracker and color checks."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capture.image_source import ImageSource
from capture.video_capture import VideoCaptureSource
from color.color_analyzer import ColorAnalyzer
from inference.inference_manager import InferenceManager
from inference.mmdet_backend import Detection
from inference.model_registry import ModelRegistry
from tracking.object_tracker import ObjectTracker
from utils.downloader import download_file
from utils.gpu_info import resolve_device
from utils.logger import get_logger, log_environment, setup_logging
from utils.paths import datasets_dir, ensure_dirs, screenshots_dir
from visualization.renderer import render

log = get_logger("smoke")

DEMO_URL = "https://raw.githubusercontent.com/open-mmlab/mmdetection/main/demo/demo.jpg"
EXPECTED_COLORS = {
    "RED": (0, 0, 220),
    "DARK RED": (0, 0, 70),
    "ORANGE": (0, 110, 230),
    "YELLOW": (0, 220, 230),
    "GREEN": (0, 180, 20),
    "DARK GREEN": (0, 60, 10),
    "CYAN": (210, 210, 20),
    "BLUE": (210, 20, 20),
    "DARK BLUE": (70, 0, 0),
    "PURPLE": (140, 20, 130),
    "PINK": (180, 170, 255),
    "WHITE": (245, 245, 245),
    "BLACK": (8, 8, 8),
    "GRAY": (130, 130, 130),
    "BROWN": (25, 55, 110),
    "BEIGE": (190, 215, 230),
}


def _solid(bgr) -> np.ndarray:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:, :] = bgr
    return image


def check_colors() -> None:
    analyzer = ColorAnalyzer()
    for name, bgr in EXPECTED_COLORS.items():
        measured = analyzer.measure(_solid(bgr), (0, 0, 80, 80), palette=False)
        got = measured["name"] if measured else None
        if got != name:
            raise SystemExit(f"Color {name} was classified as {got}")
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    image[:, :] = (255, 0, 0)
    cv2.circle(image, (60, 60), 28, (0, 0, 220), -1)
    mask = np.zeros((120, 120), dtype=np.uint8)
    cv2.circle(mask, (60, 60), 28, 1, -1)
    measured = analyzer.measure(image, (10, 10, 110, 110), mask=mask)
    if measured["name"] != "RED":
        raise SystemExit(f"Masked red circle was classified as {measured['name']}")
    print("Color checks passed")


def check_tracker() -> None:
    tracker = ObjectTracker(lost_frames_timeout=20)

    def det(x, score=0.9):
        return Detection(
            bbox=np.array([x, 10, x + 40, 50], dtype=np.float32),
            score=score,
            label=0,
            class_name="cup",
        )

    first = tracker.update([det(10)], high_thr=0.4)
    second = tracker.update([det(18)], high_thr=0.4)
    if not first or not second or first[0].track_id != second[0].track_id:
        raise SystemExit("Tracker did not keep the id while the cup moved")
    kept = second[0].track_id
    for _ in range(5):
        tracker.update([], high_thr=0.4)
    back = tracker.update([det(24)], high_thr=0.4)
    if not back or back[0].track_id != kept:
        raise SystemExit("Tracker lost the id after a short gap")
    for _ in range(25):
        tracker.update([], high_thr=0.4)
    newborn = tracker.update([det(24)], high_thr=0.4)
    if not newborn or newborn[0].track_id == kept:
        raise SystemExit("Tracker kept an id past the lost-frame timeout")
    print("Tracker checks passed")


def check_media() -> None:
    image_path = datasets_dir() / "smoke_color.png"
    video_path = datasets_dir() / "smoke_motion.avi"
    canvas = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(canvas, (40, 40), (140, 140), (0, 0, 200), -1)
    cv2.imwrite(str(image_path), canvas)
    source = ImageSource()
    source.open(str(image_path))
    ok, frame, _meta = source.read()
    if not ok or frame is None:
        raise SystemExit("Image source failed")
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (320, 240),
    )
    if not writer.isOpened():
        raise SystemExit("Could not create a test video")
    for index in range(6):
        shot = canvas.copy()
        cv2.rectangle(shot, (40 + index * 8, 40), (140 + index * 8, 140), (0, 0, 200), -1)
        writer.write(shot)
    writer.release()
    video = VideoCaptureSource()
    video.open(str(video_path))
    ok, frame, meta = video.read()
    video.close()
    if not ok or frame is None:
        raise SystemExit("Video source failed")
    rendered = render(
        frame,
        [
            {
                "track_id": 1,
                "select_key": 1,
                "class_name": "cup",
                "score": 0.91,
                "bbox": [40, 40, 140, 140],
                "mask": None,
                "color_name": "RED",
                "rgb": (200, 0, 0),
                "hsv": (0, 255, 200),
                "lab": (80, 160, 140),
                "palette": [],
                "trail": [(60, 90), (70, 90)],
            }
        ],
        {
            "detection": True,
            "segmentation": False,
            "labels": True,
            "show_confidence": True,
            "show_id": True,
            "color": True,
            "trails": True,
            "locked_id": 1,
            "locked_caption": "TARGET: CUP #1",
            "selected_id": 1,
        },
    )
    out = screenshots_dir() / "smoke_overlay.jpg"
    cv2.imwrite(str(out), rendered)
    print(f"Media checks passed ({meta.get('source')})")


def check_mmdet() -> None:
    import torch
    from mmdet.apis import DetInferencer

    import io
    from contextlib import redirect_stdout

    print("DetInferencer import: OK")
    with redirect_stdout(io.StringIO()):
        names = list(DetInferencer.list_models("mmdet") or [])
    flat = [str(name).split("::")[-1] for name in names]
    required = [
        "rtmdet_tiny_8xb32-300e_coco",
        "rtmdet_s_8xb32-300e_coco",
        "rtmdet_l_8xb32-300e_coco",
        "rtmdet-ins_tiny_8xb32-300e_coco",
        "mask-rcnn_r50_fpn_1x_coco",
    ]
    missing = [name for name in required if name not in flat]
    if missing:
        raise SystemExit(f"Installed MMDetection is missing models: {missing}")
    print(f"Model aliases present: {len(required)}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to PyTorch. The RTX GPU was not enabled.")
    device = resolve_device("auto")
    print(f"AUTO device: {device} | {torch.cuda.get_device_name(int(device.split(':')[1]))}")
    registry = ModelRegistry()
    spec = registry.get("rtmdet_tiny_8xb32-300e_coco")
    if not registry.is_installed(spec):
        print("Downloading RTMDet Tiny...")
        download_file(spec.weights_url, spec.checkpoint_path())
    ok, message = registry.verify(spec)
    if not ok:
        raise SystemExit(f"Checkpoint verify failed: {message}")
    print("Checkpoint:", message)
    image_path = datasets_dir() / "demo.jpg"
    if not image_path.exists() or image_path.stat().st_size < 1000:
        try:
            download_file(DEMO_URL, image_path)
        except Exception as exc:
            print(f"Demo image download failed ({exc}); using a synthetic image")
            synthetic = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.rectangle(synthetic, (80, 60), (280, 420), (40, 40, 180), -1)
            cv2.circle(synthetic, (420, 180), 70, (0, 0, 220), -1)
            cv2.imwrite(str(image_path), synthetic)
    image = cv2.imread(str(image_path))
    manager = InferenceManager(registry)
    try:
        result = manager.analyze(
            image,
            {"frame_index": 1, "source": "image"},
            {
                "model_id": spec.model_id,
                "device": device,
                "confidence": 0.4,
                "detection": True,
                "segmentation": False,
                "tracking": True,
                "color": True,
                "labels": True,
                "show_confidence": True,
                "show_id": True,
                "processing_scale": 1.0,
                "disabled_classes": [],
                "lost_frames_timeout": 20,
                "color_interval": 5,
                "trail_length": 30,
                "palette": False,
            },
        )
    finally:
        manager.shutdown()
    if result.get("error"):
        raise SystemExit(result["error"])
    if result.get("device") != device:
        raise SystemExit(f"Inference device was {result.get('device')}, expected {device}")
    print(f"Detections: {len(result['objects'])}")
    for obj in result["objects"][:8]:
        print(
            f"  {obj['class_name']} #{obj['track_id']} {obj['score'] * 100:.0f}%"
            f" {obj.get('color_name') or ''}"
        )
    shot = screenshots_dir() / "smoke_detection.jpg"
    cv2.imwrite(str(shot), result["rendered"])
    print(f"Saved {shot}")


def main() -> int:
    ensure_dirs()
    setup_logging()
    log_environment()
    import torch
    import mmcv
    import mmengine
    import mmdet
    from PySide6 import QtCore

    print("Python", sys.version.split()[0])
    print("Torch", torch.__version__, "CUDA", torch.version.cuda)
    print("MMCV", mmcv.__version__)
    print("MMEngine", mmengine.__version__)
    print("MMDetection", mmdet.__version__)
    print("PySide6", QtCore.__version__)
    import app.main_window  # noqa: F401

    check_colors()
    check_tracker()
    check_media()
    check_mmdet()
    try:
        from capture.camera_capture import list_cameras

        cameras = list_cameras(2)
        print("Cameras:", cameras if cameras else "none detected")
    except Exception as exc:
        print("Camera probe skipped:", exc)
    print("Smoke test passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
