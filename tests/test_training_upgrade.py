from __future__ import annotations

from types import SimpleNamespace

import pytest

from polybot.protocol import Telemetry
from polybot.pwm import PwmSteering
from polybot.training.config import architecture, estimate_ppo_parameters
from polybot.training.devices import resolve_device
from polybot.training.models import IncompatibleModelError, ModelMetadata, ModelRegistry, track_slug


def test_xl_parameter_count() -> None:
    assert architecture("xl") == (1024, 1024, 512)
    assert estimate_ppo_parameters(Telemetry.vector_size(12), (41, 2, 2), "xl") == 3_340_334


def test_pwm_is_digital_distributed_deterministic_and_resets() -> None:
    pwm = PwmSteering()
    output = pwm.generate(0.5, 30)
    assert set(output) <= {-1, 0, 1}
    assert output.count(1) == 15
    assert output[:10] != [1] * 10
    pwm.reset()
    assert output == pwm.generate(0.5, 30)
    assert pwm.generate(-0.5, 4) == [0, -1, 0, -1]
    assert pwm.generate(0, 4) == [0] * 4
    assert pwm.generate(1, 4) == [1] * 4


def test_registry_is_track_scoped_and_checks_schema(tmp_path) -> None:
    assert track_slug(" Summer 1 ") == "summer-1"
    registry = ModelRegistry(tmp_path)
    summer = ModelMetadata("Summer 1", "summer-1", "xl", 123)
    registry.write_metadata(summer)
    assert registry.metadata_path("Summer 1").parent != registry.metadata_path("Other").parent
    with pytest.raises(IncompatibleModelError):
        registry.assert_compatible(summer, track_name="Other", action_schema=summer.action_schema)


def test_device_resolution_without_cuda() -> None:
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    assert resolve_device("auto", torch).resolved == "cpu"
    assert resolve_device("cpu", torch).resolved == "cpu"
    with pytest.raises(RuntimeError, match="unavailable|is_available"):
        resolve_device("cuda", torch)
