"""CUDA device discovery for the studio."""

from __future__ import annotations

from utils.logger import get_logger

log = get_logger("gpu")


def torch_module():
    import torch

    return torch


def cuda_available() -> bool:
    try:
        torch = torch_module()
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def list_devices() -> list[dict]:
    """Return selectable devices. AUTO is not included; callers add it."""
    devices = [{"id": "cpu", "label": "CPU", "name": "CPU", "total": 0}]
    try:
        torch = torch_module()
    except Exception as exc:
        log.info("Torch import failed while listing devices: %s", exc)
        return devices
    if not torch.cuda.is_available():
        return devices
    found = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        name = props.name
        total = int(props.total_memory)
        found.append(
            {
                "id": f"cuda:{index}",
                "label": f"CUDA:{index} — {name}",
                "name": name,
                "total": total,
            }
        )
    found.sort(key=lambda item: item["total"], reverse=True)
    return found + devices


def resolve_device(choice: str) -> str:
    """Map AUTO / CUDA:n / CPU onto a torch device string."""
    choice = (choice or "auto").strip().lower()
    if choice in {"cpu"}:
        return "cpu"
    if choice.startswith("cuda"):
        return choice
    if not cuda_available():
        return "cpu"
    torch = torch_module()
    best_index = 0
    best_memory = -1
    for index in range(torch.cuda.device_count()):
        total = int(torch.cuda.get_device_properties(index).total_memory)
        if total > best_memory:
            best_memory = total
            best_index = index
    return f"cuda:{best_index}"


def device_status(device: str) -> dict:
    info = {
        "device": device,
        "name": "CPU",
        "total": 0,
        "used": 0,
        "free": 0,
    }
    if not device or not str(device).startswith("cuda"):
        return info
    try:
        torch = torch_module()
        index = int(str(device).split(":")[1]) if ":" in str(device) else 0
        props = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        info.update(
            {
                "name": props.name,
                "total": int(total),
                "free": int(free),
                "used": int(total - free),
            }
        )
    except Exception as exc:
        log.info("VRAM query failed: %s", exc)
    return info


def format_gb(num_bytes: int) -> str:
    if num_bytes <= 0:
        return "0.0 GB"
    return f"{num_bytes / (1024 ** 3):.1f} GB"
