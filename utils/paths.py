"""Project paths and directory setup."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def config_dir() -> Path:
    return project_root() / "config"


def settings_file() -> Path:
    return config_dir() / "settings.json"


def checkpoints_dir() -> Path:
    return project_root() / "checkpoints"


def screenshots_dir() -> Path:
    return project_root() / "screenshots"


def objects_dir() -> Path:
    return project_root() / "objects"


def logs_dir() -> Path:
    return project_root() / "logs"


def custom_models_dir() -> Path:
    return project_root() / "custom_models"


def custom_registry_file() -> Path:
    return custom_models_dir() / "registry.json"


def datasets_dir() -> Path:
    return project_root() / "datasets"


def color_ranges_file() -> Path:
    return project_root() / "color" / "color_ranges.json"


def ensure_dirs() -> None:
    for path in (
        project_root() / "models",
        checkpoints_dir(),
        screenshots_dir(),
        objects_dir(),
        datasets_dir(),
        project_root() / "training",
        custom_models_dir(),
        logs_dir(),
        config_dir(),
    ):
        path.mkdir(parents=True, exist_ok=True)
