"""File and console logging for the studio."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from utils.paths import logs_dir

_CONFIGURED = False


def setup_logging() -> logging.Logger:
    global _CONFIGURED
    root = logging.getLogger("vision")
    if _CONFIGURED:
        return root
    logs_dir().mkdir(parents=True, exist_ok=True)
    root.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        logs_dir() / "app.log",
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    root.propagate = False
    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(f"vision.{name}")


def log_environment() -> None:
    log = get_logger("env")
    log.info("Python %s", sys.version.replace("\n", " "))
    try:
        import torch

        log.info(
            "Torch %s | CUDA build %s | cuda available %s",
            torch.__version__,
            torch.version.cuda,
            torch.cuda.is_available(),
        )
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                log.info(
                    "GPU cuda:%s %s VRAM %.1f GB",
                    index,
                    props.name,
                    props.total_memory / (1024 ** 3),
                )
    except Exception as exc:
        log.info("Torch is not available: %s", exc)
    for module_name in ("mmcv", "mmengine", "mmdet"):
        try:
            module = __import__(module_name)
            log.info("%s %s", module_name, getattr(module, "__version__", "unknown"))
        except Exception as exc:
            log.info("%s is not available: %s", module_name, exc)
