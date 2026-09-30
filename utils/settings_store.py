"""Load and save config/settings.json."""

from __future__ import annotations

import json
from copy import deepcopy

from utils.logger import get_logger
from utils.paths import settings_file

log = get_logger("settings")

DEFAULT_SETTINGS = {
    "source": "webcam",
    "camera_index": 0,
    "resolution": "1280x720",
    "device": "auto",
    "profile": "fast",
    "explicit_model_id": "",
    "optimization": "balanced",
    "confidence": 0.40,
    "processing_scale": 0.75,
    "last_video": "",
    "last_image": "",
    "modes": {
        "detection": True,
        "segmentation": False,
        "tracking": True,
        "color": True,
        "trails": False,
        "labels": True,
        "confidence": True,
        "object_id": True,
        "debug": False,
        "detailed_color": False,
        "palette": False,
    },
    "disabled_classes": [],
    "lost_frames_timeout": 20,
    "color_interval": 5,
    "trail_length": 30,
    "geometry": None,
}


def _merge(base: dict, incoming: dict) -> dict:
    merged = deepcopy(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_settings() -> dict:
    path = settings_file()
    if not path.exists():
        return deepcopy(DEFAULT_SETTINGS)
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.exception("Could not read settings, using defaults")
        return deepcopy(DEFAULT_SETTINGS)
    if not isinstance(stored, dict):
        return deepcopy(DEFAULT_SETTINGS)
    return _merge(DEFAULT_SETTINGS, stored)


def save_settings(settings: dict) -> None:
    path = settings_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8")
