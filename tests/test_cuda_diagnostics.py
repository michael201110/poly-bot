from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from polybot.training import devices


def fake_torch(*, cuda_build: str | None, available: bool, name: str = "NVIDIA T500"):
    return SimpleNamespace(
        __version__="2.9.0", version=SimpleNamespace(cuda=cuda_build),
        cuda=SimpleNamespace(
            is_available=lambda: available, device_count=lambda: int(available),
            get_device_name=lambda index: name, current_device=lambda: 0,
        ),
    )


@pytest.fixture(autouse=True)
def no_external_gpu_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        devices.subprocess, "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="NVIDIA T500\n"),
    )


def test_cpu_only_build_reports_install_action() -> None:
    torch = fake_torch(cuda_build=None, available=False)
    facts = devices.cuda_diagnostics(torch)
    assert facts["nvidia_gpu"] == "NVIDIA T500"
    assert facts["torch_cuda_build"] is None
    assert devices.resolve_device("auto", torch).diagnostics["selection_reason"].startswith(
        "NVIDIA GPU"
    )
    with pytest.raises(RuntimeError, match="CPU-only PyTorch build"):
        devices.resolve_device("cuda", torch)


def test_cuda_build_without_initialized_gpu_fails_explicit_selection() -> None:
    torch = fake_torch(cuda_build="12.6", available=False)
    assert devices.resolve_device("auto", torch).resolved == "cpu"
    with pytest.raises(RuntimeError, match="CUDA initialization failed"):
        devices.resolve_device("cuda", torch)


def test_cuda_available_selects_named_device() -> None:
    torch = fake_torch(cuda_build="12.6", available=True)
    selected = devices.resolve_device("auto", torch)
    assert selected.resolved == "cuda"
    assert selected.gpu_name == "NVIDIA T500"
    assert selected.diagnostics["cuda_device_count"] == 1
    assert devices.resolve_device("cpu", torch).resolved == "cpu"


def test_optional_cuda_calls_can_fail_without_crashing() -> None:
    def broken(*args):
        raise RuntimeError("driver failed")

    torch = SimpleNamespace(
        __version__="2.9.0", version=SimpleNamespace(cuda="12.6"),
        cuda=SimpleNamespace(is_available=lambda: True, device_count=broken,
                             get_device_name=broken, current_device=broken),
        backends=SimpleNamespace(cudnn=SimpleNamespace(is_available=broken, version=broken)),
    )
    facts = devices.cuda_diagnostics(torch)
    assert facts["cuda_device_count"] == 0
    assert facts["cuda_device_name"] is None
    assert facts["cudnn_available"] is None
    assert devices.resolve_device("auto", torch).resolved == "cpu"
    with pytest.raises(RuntimeError, match="CUDA initialization failed"):
        devices.resolve_device("cuda", torch)


def test_torch_missing_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = __import__

    def missing_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", missing_torch)
    assert devices.cuda_diagnostics()["torch_version"] is None
    with pytest.raises(RuntimeError, match="PyTorch is not installed"):
        devices.resolve_device("cuda")


def test_smoke_device_check_accepts_cuda_index_and_rejects_wrong_type() -> None:
    cuda_parameter = SimpleNamespace(device=torch.device("cuda:0"))
    cpu_parameter = SimpleNamespace(device=torch.device("cpu"))
    assert devices.checked_parameter_device(cuda_parameter, "cuda") == "cuda:0"
    assert devices.checked_parameter_device(cpu_parameter, "cpu") == "cpu"
    with pytest.raises(RuntimeError, match="expected a cuda device"):
        devices.checked_parameter_device(cpu_parameter, "cuda")
