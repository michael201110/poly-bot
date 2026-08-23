"""Torch device selection kept independently testable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    requested: str
    resolved: str
    gpu_name: str | None = None


def resolve_device(requested: str, torch_module: Any | None = None) -> DeviceInfo:
    requested = requested.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:
            if requested == "cuda":
                raise RuntimeError("CUDA was requested, but PyTorch is not installed") from None
            return DeviceInfo(requested, "cpu")
    available = bool(torch_module.cuda.is_available())
    if requested == "cuda" and not available:
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    resolved = "cuda" if available and requested != "cpu" else "cpu"
    name = str(torch_module.cuda.get_device_name(0)) if resolved == "cuda" else None
    return DeviceInfo(requested, resolved, name)
