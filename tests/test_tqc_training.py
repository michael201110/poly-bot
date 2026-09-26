"""TQC control and persistence tests that run without a GPU or live game."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest
from gymnasium import spaces

from polybot.env import ControlDuty, PolyTrackEnv, RewardConfig, _expert_action_reward
from polybot.mock import MockSimulatorTransport
from polybot.pwm import ContinuousPwmControls
from polybot.training.config import (
    TqcConfig,
    TrainingConfig,
    estimate_tqc_actor_parameters,
    estimate_tqc_parameters,
)
from polybot.training.models import IncompatibleModelError, ModelMetadata, ModelRegistry
from polybot.training.trainer import (
    ScaledTrainingReward,
    TrainingService,
    tqc_policy_diagnostics,
)


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


def test_continuous_reward_uses_duty_and_tracks_applied_pulses() -> None:
    rewards = RewardConfig(
        ground_brake_penalty_per_s=-100.0,
        action_change_penalty=-10.0,
    )

    def step(longitudinal: float):
        env = PolyTrackEnv(
            MockSimulatorTransport(), track_id="mock/straight", frame_skip=16,
            action_mode="continuous_pwm", reward_config=rewards,
        )
        try:
            env.reset(seed=3)
            return env.step(np.array([0.0, longitudinal], dtype=np.float32))[4]
        finally:
            env.close()

    tiny_brake = step(-0.01)
    full_brake = step(-1.0)
    assert tiny_brake["reward_terms"]["ground_brake"] == pytest.approx(
        full_brake["reward_terms"]["ground_brake"] * 0.01, rel=1e-5
    )
    assert tiny_brake["reward_terms"]["action_change"] == pytest.approx(-0.1, rel=1e-5)
    assert full_brake["reward_terms"]["action_change"] == pytest.approx(-10.0)
    assert tiny_brake["requested_control_duty"]["brake"] == pytest.approx(0.01)
    assert tiny_brake["applied_control_fraction"]["brake"] == 0.0
    quarter_throttle = step(0.25)
    full_throttle = step(1.0)
    assert quarter_throttle["requested_control_duty"]["throttle"] == 0.25
    assert quarter_throttle["applied_control_fraction"]["throttle"] == pytest.approx(0.25)
    assert full_throttle["applied_control_fraction"]["throttle"] == 1.0
    assert quarter_throttle["reward_terms"]["action_change"] == pytest.approx(-2.5)
    assert full_throttle["reward_terms"]["action_change"] == pytest.approx(-10.0)


def test_tiny_tqc_parameter_counts_match_actual_model() -> None:
    from polybot.training.algorithms import create_model

    env = PolyTrackEnv(MockSimulatorTransport(), action_mode="continuous_pwm")
    try:
        cfg = TrainingConfig(algorithm="tqc", tqc=TqcConfig(architecture="tiny", buffer_size=64))
        model = create_model(cfg, env, "cpu")
        assert estimate_tqc_actor_parameters(105, 2, "tiny") == 11_204
        assert estimate_tqc_parameters(105, 2, "tiny") == 61_992
        assert sum(p.numel() for p in model.policy.actor.parameters()) == 11_204
        assert sum(p.numel() for p in model.policy.parameters()) == 61_992
    finally:
        env.close()


def test_tqc_warmup_prefers_forward_actions_and_actor_starts_forward() -> None:
    from polybot.training.algorithms import create_model

    env = PolyTrackEnv(MockSimulatorTransport(), action_mode="continuous_pwm")
    try:
        cfg = TrainingConfig(
            algorithm="tqc", seed=3,
            tqc=TqcConfig(architecture="tiny", buffer_size=64),
        )
        model = create_model(cfg, env, "cpu")
        actions = np.array([model._sample_action(5_000)[0][0] for _ in range(200)])
        assert (actions[:, 1] >= 0.65).sum() >= 140
        assert np.median(np.abs(actions[:, 0])) < 0.6
        assert model.policy.actor.mu.bias[1].item() > 0.8
        deterministic, _ = model.predict(np.zeros(105, dtype=np.float32), deterministic=True)
        assert deterministic[1] > 0.5
        probe = tqc_policy_diagnostics(model, np.zeros(105, dtype=np.float32))
        assert probe["deterministic_longitudinal"] == pytest.approx(deterministic[1])
        assert set(probe["critic_longitudinal_q"]) == {"1.0", "0.5", "0.0", "-0.5"}
        model.num_timesteps = 5_000
        model._last_obs = np.zeros((1, 105), dtype=np.float32)
        # Once learning starts, the standard stochastic TQC actor samples actions.
        action, buffered = model._sample_action(5_000)
        assert action.shape == buffered.shape == (1, 2)
    finally:
        env.close()


def test_tqc_warmup_seed_and_replay_action_match_execution() -> None:
    from polybot.training.algorithms import create_model

    samples = []
    for _ in range(2):
        env = PolyTrackEnv(MockSimulatorTransport(), track_id="mock/straight",
                           frame_skip=16, action_mode="continuous_pwm")
        try:
            model = create_model(TrainingConfig(algorithm="tqc", seed=19,
                                  tqc=TqcConfig(architecture="tiny", buffer_size=64)), env, "cpu")
            env.reset(seed=19)
            run = []
            for _ in range(12):
                executed, buffered = model._sample_action(5_000)
                np.testing.assert_allclose(model.policy.unscale_action(buffered), executed)
                _, _, terminated, truncated, info = env.step(executed[0])
                requested = info["requested_control_duty"]
                assert requested["throttle"] == pytest.approx(max(0, executed[0, 1]))
                assert requested["brake"] == pytest.approx(max(0, -executed[0, 1]))
                assert not (requested["throttle"] and requested["brake"])
                run.append(executed[0].copy())
                if terminated or truncated:
                    env.reset(seed=19)
            samples.append(run)
        finally:
            env.close()
    np.testing.assert_array_equal(samples[0], samples[1])


def test_tqc_forward_prior_is_annealed_and_buffered_action_is_executed() -> None:
    import torch

    from polybot.training.algorithms import create_model

    env = PolyTrackEnv(MockSimulatorTransport(), action_mode="continuous_pwm")
    try:
        cfg = TrainingConfig(
            algorithm="tqc", tqc=TqcConfig(
                architecture="tiny", buffer_size=64,
                forward_prior_initial=0.7, forward_prior_steps=100,
            ),
        )
        model = create_model(cfg, env, "cpu")
        model._last_obs = np.zeros((1, 105), dtype=np.float32)
        model.num_timesteps = 5_000
        assert model.forward_prior_strength(5_000) == pytest.approx(0.7)
        torch.manual_seed(5)
        model.forward_prior_initial = 0.0
        baseline, _ = model._sample_action(5_000)
        model.forward_prior_initial = 0.7
        torch.manual_seed(5)
        executed, buffered = model._sample_action(5_000)
        assert executed[0, 1] == pytest.approx(min(1.0, baseline[0, 1] + 0.7))
        np.testing.assert_allclose(model.policy.unscale_action(buffered), executed)
        model.num_timesteps = 5_050
        assert model.forward_prior_strength(5_000) == pytest.approx(0.35)
        model.num_timesteps = 5_100
        assert model.forward_prior_strength(5_000) == 0.0
        with torch.no_grad():
            model.policy.actor.mu.bias[1] = -2.0
        actions = [model._sample_action(5_000)[0][0, 1] for _ in range(20)]
        assert any(value < 0 for value in actions)
    finally:
        env.close()


def test_ppo_brake_reward_keeps_binary_behavior() -> None:
    rewards = RewardConfig(
        ground_brake_penalty_per_s=-100.0, action_change_penalty=-10.0
    )
    env = PolyTrackEnv(
        MockSimulatorTransport(), track_id="mock/straight", frame_skip=16,
        action_mode="ppo_pwm", reward_config=rewards,
    )
    try:
        env.reset(seed=3)
        info = env.step(np.array([20, 0, 1], dtype=np.int64))[4]
        dt = info["ticks_advanced"] * env.simulator_capabilities["fixed_dt_s"]
        assert info["reward_terms"]["ground_brake"] == pytest.approx(-100.0 * dt)
        assert info["reward_terms"]["action_change"] == pytest.approx(-10.0)
        assert "requested_control_duty" not in info
    finally:
        env.close()


def test_continuous_expert_similarity_is_continuous_at_zero() -> None:
    env = PolyTrackEnv(MockSimulatorTransport(), track_id="mock/straight")
    try:
        env.reset(seed=3)
        telemetry = env.latest_telemetry
        assert telemetry is not None
        rewards = RewardConfig(expert_action_bonus_per_s=90.0)
        values = [
            _expert_action_reward(
                ControlDuty.from_continuous(0.0, longitudinal), telemetry, rewards, 0.1
            )
            for longitudinal in (-0.01, 0.0, 0.01)
        ]
        assert abs(values[0] - values[1]) < 0.1
        assert abs(values[2] - values[1]) < 0.1
    finally:
        env.close()


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


def test_successful_lap_rehearsal_updates_actor_and_survives_save(tmp_path) -> None:
    import torch

    from polybot.training.algorithms import create_model
    from polybot.training.forward_tqc import ForwardWarmupTQC

    env = PolyTrackEnv(MockSimulatorTransport(), action_mode="continuous_pwm")
    try:
        model = create_model(
            TrainingConfig(algorithm="tqc", tqc=TqcConfig(architecture="tiny", buffer_size=64)),
            env, "cpu",
        )
        observation, _ = env.reset(seed=4)
        observations = np.repeat(observation[None], 32, axis=0)
        actions = np.repeat(np.array([[-0.5, -0.5]], dtype=np.float32), 32, axis=0)
        model.remember_successful_trajectory(observations, actions)
        with torch.no_grad():
            before = torch.nn.functional.mse_loss(
                model.actor(torch.as_tensor(observations), deterministic=True),
                torch.as_tensor(actions),
            ).item()
        for _ in range(20):
            model._rehearse_success(32)
        with torch.no_grad():
            after = torch.nn.functional.mse_loss(
                model.actor(torch.as_tensor(observations), deterministic=True),
                torch.as_tensor(actions),
            ).item()
        assert after < before
        archive = tmp_path / "rehearsal.zip"
        model.save(str(archive))
        restored = ForwardWarmupTQC.load(str(archive), device="cpu")
        assert len(restored.successful_trajectories) == 1
        np.testing.assert_array_equal(restored.successful_trajectories[0][1], actions)
    finally:
        env.close()
