"""Safe, independently testable PyTorch CUDA diagnostics and device selection."""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class DeviceInfo:
    requested: str
    resolved: str
    gpu_name: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _safe_call(function: Any, default: Any = None) -> Any:
    try:
        return function()
    except Exception:
        return default


def cuda_diagnostics(torch_module: Any | None = None) -> dict[str, Any]:
    """Return CUDA facts without assuming optional CUDA APIs can initialize."""

    probe_driver = torch_module is None
    result: dict[str, Any] = {
        "python": sys.executable,
        "platform": platform.platform(),
        "torch_version": None,
        "torch_cuda_build": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_name": None,
        "cuda_current_device": None,
        "cudnn_available": None,
        "cudnn_version": None,
        "nvidia_gpu": None,
        "cuda_driver_init_code": None,
    }
    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:
            torch_module = None
    if torch_module is not None:
        result["torch_version"] = str(getattr(torch_module, "__version__", "unknown"))
        result["torch_cuda_build"] = getattr(getattr(torch_module, "version", None), "cuda", None)
        cuda = getattr(torch_module, "cuda", None)
        if cuda is not None:
            available_method = getattr(cuda, "is_available", lambda: False)
            result["cuda_available"] = bool(_safe_call(available_method, False))
            result["cuda_device_count"] = int(
                _safe_call(getattr(cuda, "device_count", lambda: 0), 0) or 0
            )
            if result["cuda_available"]:
                result["cuda_device_name"] = _safe_call(
                    lambda: cuda.get_device_name(0)
                )
                result["cuda_current_device"] = _safe_call(
                    getattr(cuda, "current_device", lambda: None)
                )
        cudnn = getattr(getattr(torch_module, "backends", None), "cudnn", None)
        if cudnn is not None:
            result["cudnn_available"] = _safe_call(cudnn.is_available)
            result["cudnn_version"] = _safe_call(cudnn.version)
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=3, check=False,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            result["nvidia_gpu"] = probe.stdout.strip().splitlines()[0]
    except (OSError, subprocess.TimeoutExpired):
        pass
    if (
        probe_driver and sys.platform == "win32" and result["nvidia_gpu"]
        and result["torch_cuda_build"] and not result["cuda_available"]
    ):
        try:
            result["cuda_driver_init_code"] = ctypes.WinDLL("nvcuda.dll").cuInit(0)
        except (AttributeError, OSError):
            pass
    return result


def _unavailable_reason(diagnostics: dict[str, Any]) -> str:
    if diagnostics["torch_version"] is None:
        return "PyTorch is not installed; install PolyBot's train extra first."
    if diagnostics["torch_cuda_build"] is None:
        gpu = diagnostics["nvidia_gpu"]
        return (
            f"NVIDIA GPU {gpu} is present, but this is a CPU-only PyTorch build. "
            "Install a CUDA-enabled PyTorch wheel from the official package index; "
            "see docs/training.md."
            if gpu else (
                "This PyTorch build is CPU-only; see docs/training.md for CUDA installation."
            )
        )
    if diagnostics["nvidia_gpu"]:
        driver_code = diagnostics.get("cuda_driver_init_code")
        detail = (
            " The CUDA driver API reports NO_DEVICE (100), despite nvidia-smi seeing the GPU."
            if driver_code == 100 else (
                f" The CUDA driver API returned {driver_code}."
                if driver_code is not None else ""
            )
        )
        return (
            f"NVIDIA GPU {diagnostics['nvidia_gpu']} is detected and PyTorch has CUDA "
            f"{diagnostics['torch_cuda_build']}, but CUDA initialization failed.{detail} "
            "Check the NVIDIA driver and run polybot-doctor for details."
        )
    return "No usable NVIDIA/CUDA device was found."


def resolve_device(requested: str, torch_module: Any | None = None) -> DeviceInfo:
    requested = requested.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    facts = cuda_diagnostics(torch_module)
    if requested == "cuda" and (
        not facts["cuda_available"] or facts["cuda_device_count"] < 1
    ):
        raise RuntimeError(_unavailable_reason(facts))
    if requested != "cpu" and facts["cuda_available"] and facts["cuda_device_count"] > 0:
        if not facts["cuda_device_name"]:
            if requested == "cuda":
                raise RuntimeError("CUDA is reported available, but GPU initialization failed.")
            facts["selection_reason"] = "GPU name lookup failed; using CPU"
        else:
            facts["selection_reason"] = "CUDA initialized successfully"
            return DeviceInfo(requested, "cuda", facts["cuda_device_name"], facts)
    if "selection_reason" not in facts:
        facts["selection_reason"] = (
            "CPU explicitly selected" if requested == "cpu" else _unavailable_reason(facts)
        )
    return DeviceInfo(requested, "cpu", diagnostics=facts)


def doctor_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show PolyBot PyTorch/CUDA diagnostics")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke-tqc", action="store_true")
    args = parser.parse_args(argv)
    try:
        selected = resolve_device(args.device)
    except RuntimeError as exc:
        facts = cuda_diagnostics()
        facts["error"] = str(exc)
        print(json.dumps(facts, indent=2))
        return 2
    facts = selected.diagnostics
    facts["selected_device"] = selected.resolved
    if args.smoke_tqc:
        from polybot.env import PolyTrackEnv
        from polybot.mock import MockSimulatorTransport
        from polybot.training.algorithms import create_model
        from polybot.training.config import TrainingConfig

        config = TrainingConfig(algorithm="tqc", backend="mock", device=selected.resolved)
        env = PolyTrackEnv(
            MockSimulatorTransport(), track_id="mock/straight", action_mode="continuous_pwm"
        )
        try:
            model = create_model(config, env, selected.resolved)
            facts["tqc_parameter_device"] = str(next(model.policy.parameters()).device)
            if facts["tqc_parameter_device"] != selected.resolved:
                raise RuntimeError("TQC parameters were placed on the wrong device")
        finally:
            env.close()
    print(json.dumps(facts, indent=2))
    return 0
