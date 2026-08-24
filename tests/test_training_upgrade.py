from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from polybot.env import (
    PolyTrackEnv,
    summer_1_pace_reward_config,
    summer_1_reward_config,
)
from polybot.mock import MockSimulatorTransport
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


def test_pwm_sequence_uses_one_protocol_round_trip() -> None:
    transport = MockSimulatorTransport()
    env = PolyTrackEnv(
        transport,
        track_id="mock/gentle-s",
        pwm_enabled=True,
        pwm_levels=41,
        frame_skip=16,
    )
    env.reset(seed=1)
    before = len(transport.command_log)
    env.step(np.array([30, 1, 0]))  # +50% steering
    requests = transport.command_log[before:]
    assert len(requests) == 1
    params = requests[0]["params"]
    assert len(params["actions"]) == 16
    assert {action["steer"] for action in params["actions"]} == {0.0, 1.0}
    env.close()


def test_summer_1_reward_profile_is_dense_and_balanced() -> None:
    rewards = summer_1_reward_config()
    assert rewards.progress_per_m > 0
    assert rewards.on_track_speed_per_m > 0
    assert rewards.checkpoint_bonus > 0
    assert rewards.finish_bonus > rewards.checkpoint_bonus
    assert rewards.crash_penalty < 0
    assert rewards.off_track_penalty < 0
    assert -100 < rewards.ground_slip_penalty_per_rad_s < 0


def test_fresh_run_archive_preserves_latest_and_metadata(tmp_path) -> None:
    registry = ModelRegistry(tmp_path)
    directory = registry.initialise_track("Summer 1")
    (directory / "latest.zip").write_bytes(b"model")
    (directory / "latest.metadata.json").write_text("{}", encoding="utf-8")
    archived = registry.archive_latest("Summer 1")
    assert archived is not None
    assert archived.read_bytes() == b"model"
    assert archived.with_suffix(".metadata.json").exists()
    assert (directory / "latest.zip").read_bytes() == b"model"


def test_pace_profile_increases_time_pressure_without_removing_safety() -> None:
    balanced = summer_1_reward_config()
    pace = summer_1_pace_reward_config()
    assert pace.elapsed_cost_per_s < balanced.elapsed_cost_per_s
    assert pace.checkpoint_fast_bonus > balanced.checkpoint_fast_bonus
    assert pace.checkpoint_target_s < balanced.checkpoint_target_s
    assert pace.checkpoint_speed_bonus_per_mps > 0
    assert pace.imitation_bonus_per_s < balanced.imitation_bonus_per_s
    assert pace.crash_penalty == balanced.crash_penalty
    assert pace.off_track_penalty == balanced.off_track_penalty
