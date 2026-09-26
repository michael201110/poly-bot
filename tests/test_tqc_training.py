"""TQC control and persistence tests that run without a GPU or live game."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest
from gymnasium import spaces

from polybot.env import PolyTrackEnv
from polybot.mock import MockSimulatorTransport
from polybot.pwm import ContinuousPwmControls
from polybot.training.config import TqcConfig, TrainingConfig, estimate_tqc_parameters
from polybot.training.models import IncompatibleModelError, ModelMetadata, ModelRegistry
from polybot.training.trainer import ScaledTrainingReward, TrainingService


def test_continuous_pwm_duties_and_reset() -> None:
    controls = ContinuousPwmControls()
    assert controls.generate(1, 1, 10) == [(1, True, False)] * 10
    assert controls.generate(-1, -1, 10) == [(-1, False, True)] * 10
    assert controls.generate(0, 0, 10) == [(0, False, False)] * 10
    sequence = controls.generate(0.25, 0.75, 20)
    assert sum(steer == 1 for steer, _, _ in sequence) == 5
    assert sum(throttle for _, throttle, _ in sequence) == 15
    assert not any(throttle and brake for _, throttle, brake in sequence)
    braking = controls.generate(0, -0.4, 20)
    assert sum(brake for _, _, brake in braking) == 8
    assert not any(throttle and brake for _, throttle, brake in braking)
    controls.reset()
    first = controls.generate(0.25, 0.75, 3)
    second = controls.generate(0.25, 0.75, 3)
    assert sum(steer == 1 for steer, _, _ in first) == 0
    assert sum(steer == 1 for steer, _, _ in second) == 1
    controls.reset()
    assert controls.generate(0.25, 0.75, 3) == first
    replica = ContinuousPwmControls()
    assert replica.generate(0.25, 0.75, 20) == sequence


def test_env_action_spaces_and_wire_sequence() -> None:
    transport = MockSimulatorTransport()
    continuous = PolyTrackEnv(
        transport, track_id="mock/straight", frame_skip=16, action_mode="continuous_pwm"
    )
    assert isinstance(continuous.action_space, spaces.Box)
    np.testing.assert_array_equal(continuous.action_space.low, [-1, -1])
    np.testing.assert_array_equal(continuous.action_space.high, [1, 1])
    continuous.reset(seed=3)
    continuous.step(np.array([0.25, 0.75], dtype=np.float32))
    actions = transport.command_log[-1]["params"]["actions"]
    assert len(actions) == 16
    assert sum(action["steer"] == 1 for action in actions) == 4
    assert sum(action["throttle"] == 1 for action in actions) == 12
    assert all(not (action["throttle"] and action["brake"]) for action in actions)
    continuous.close()
    ppo = PolyTrackEnv(MockSimulatorTransport(), pwm_enabled=True, pwm_levels=41)
    digital = PolyTrackEnv(MockSimulatorTransport())
    np.testing.assert_array_equal(ppo.action_space.nvec, [41, 2, 2])
    np.testing.assert_array_equal(digital.action_space.nvec, [3, 2, 2])
    assert ppo.observation_space.shape == continuous.observation_space.shape
    ppo.close()
    digital.close()


def test_tqc_creation_save_resume_and_raw_reward_logging(tmp_path) -> None:
    events: list[dict] = []
    config = TrainingConfig(
        algorithm="tqc", backend="mock", track_name="Mock Straight",
        track_id="mock/straight", frame_skip=8, max_episode_steps=8,
        timesteps=32, output_root=tmp_path, device="cpu", checkpoint_interval=0,
        tqc=TqcConfig(architecture="compact", buffer_size=128, learning_starts=2,
                      batch_size=8),
    )
    service = TrainingService(config, events.append)
    path = service.run()
    assert path == tmp_path / "mock-straight" / "tqc" / "latest.zip"
    assert path.with_suffix(".replay.pkl").is_file()
    metadata = ModelRegistry.metadata_for_archive(path)
    assert metadata.algorithm == "TQC"
    assert metadata.action_schema == "continuous-pwm-v1"
    assert metadata.simulator_ticks > 0
    assert metadata.tqc_hyperparameters == asdict(config.tqc)
    assert metadata.ppo_hyperparameters == {}
    assert metadata.parameter_count == estimate_tqc_parameters(105, 2, "compact")
    assert any(event["type"] == "episode" for event in events)
    for episode in (event for event in events if event["type"] == "episode"):
        assert episode["reward"] == pytest.approx(sum(episode["reward_terms"].values()))
    second = TrainingService(config, events.append).run(resume=path)
    assert second == path
    resumed = ModelRegistry.metadata_for_archive(path)
    assert resumed.training_timesteps == 64
    assert resumed.simulator_ticks > metadata.simulator_ticks
    assert resumed.training_episodes > metadata.training_episodes
    assert resumed.wall_clock_seconds > metadata.wall_clock_seconds
    changed = TrainingConfig(
        algorithm="tqc", backend="mock", track_name="Mock Straight",
        track_id="mock/straight", output_root=tmp_path, device="cpu",
        tqc=TqcConfig(architecture="compact", buffer_size=256, learning_starts=2,
                      batch_size=8),
    )
    with pytest.raises(IncompatibleModelError, match="settings differ"):
        TrainingService(changed).run(resume=path)


def test_reward_wrapper_scales_both_algorithms_without_changing_raw_terms() -> None:
    for mode, action in (
        ("ppo_pwm", np.array([20, 1, 0])),
        ("continuous_pwm", np.array([0.0, 1.0], dtype=np.float32)),
    ):
        env = ScaledTrainingReward(
            PolyTrackEnv(
                MockSimulatorTransport(), track_id="mock/straight",
                frame_skip=8, action_mode=mode,
            ),
            0.01,
        )
        env.reset(seed=4)
        _, scaled, _, _, info = env.step(action)
        assert scaled == pytest.approx(sum(info["reward_terms"].values()) * 0.01)
        env.close()


def test_algorithm_scoped_models_preserve_legacy_ppo(tmp_path) -> None:
    registry = ModelRegistry(tmp_path)
    legacy = registry.initialise_track("Summer 1") / "latest.zip"
    legacy.write_bytes(b"old PPO")
    old = ModelMetadata("Summer 1", "current", "compact", 100)
    registry.write_metadata(old, "latest")
    new = registry.initialise_track("Summer 1", "tqc") / "latest.zip"
    new.write_bytes(b"new TQC")
    tqc = ModelMetadata("Summer 1", "current", "standard", 200,
                        algorithm="TQC", action_schema="continuous-pwm-v1")
    registry.write_metadata(tqc, "latest", "tqc")
    assert legacy.read_bytes() == b"old PPO"
    assert registry.list_models("Summer 1", "ppo") == [legacy]
    assert registry.list_models("Summer 1", "tqc") == [new]
    registry.assert_compatible(old, track_name="Summer 1",
                               action_schema="pwm-multidiscrete-v1", algorithm="ppo")
    with pytest.raises(IncompatibleModelError, match="algorithm"):
        registry.assert_compatible(tqc, track_name="Summer 1",
                                   action_schema="continuous-pwm-v1", algorithm="ppo")
    legacy_metadata = json.loads(registry.metadata_path("Summer 1", "latest").read_text())
    assert legacy_metadata["algorithm"] == "PPO"


def test_tqc_model_is_on_cuda_when_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    from polybot.training.algorithms import create_model

    env = PolyTrackEnv(MockSimulatorTransport(), action_mode="continuous_pwm")
    try:
        model = create_model(TrainingConfig(algorithm="tqc", tqc=TqcConfig(buffer_size=64)),
                             env, "cuda")
        assert next(model.policy.parameters()).is_cuda
    finally:
        env.close()
