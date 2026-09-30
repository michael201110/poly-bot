from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.torch_layers import FlattenExtractor
from torch import nn

from polybot.algorithms.ppo_tqc import initialize_actor_from_tqc
from polybot.algorithms.registry import backend_for
from polybot.environment.env import AirBrakeActionWrapper, PolyTrackEnv
from polybot.mock import MockSimulatorTransport
from polybot.training.compare_teacher_student import (
    _record,
    _transformed_deterministic_mode,
    first_divergences,
    replay_actions,
)
from polybot.training.config import PPOConfig, TQCConfig, TrainingConfig


def test_first_divergences_reports_threshold_crossings() -> None:
    teacher = [
        {"action": [0.0, 1.0], "position_m": [0, 0, 0], "heading_deg": 0.0,
         "wheel_contacts": [1, 1, 1, 1], "speed_mps": 10.0,
         "steering": 0.0, "longitudinal": 1.0},
        {"action": [0.0, 1.0], "position_m": [1, 0, 0], "heading_deg": 0.0,
         "wheel_contacts": [1, 1, 1, 1], "speed_mps": 10.0,
         "steering": 0.0, "longitudinal": 1.0},
    ]
    student = [
        {**teacher[0], "action": [0.006, 1.0], "steering": 0.006},
        {**teacher[1], "position_m": [1.11, 0, 0], "heading_deg": 0.6,
         "wheel_contacts": [0, 1, 1, 1], "speed_mps": 9.9},
    ]
    divergence = first_divergences(teacher, student)
    assert divergence["action"]["0.005"] == 0
    assert divergence["action"]["0.01"] is None
    assert divergence["position_m"]["0.1"] == 1
    assert divergence["heading_deg"]["0.5"] == 1
    assert divergence["wheel_contacts"] == 1


def test_identical_continuous_actions_replay_identically_with_both_backends() -> None:
    action = np.asarray([0.137, 0.684], dtype=np.float32)
    tick_sequences = []
    for algorithm, kwargs in (("tqc", {"tqc": TQCConfig()}),
                              ("ppo", {"ppo": PPOConfig()})):
        config = TrainingConfig(algorithm=algorithm, **kwargs)
        adapter = backend_for(algorithm).action_adapter(config)
        env = PolyTrackEnv(MockSimulatorTransport(), track_id="mock/straight",
                           frame_skip=30, action_adapter=adapter)
        env.capture_tick_controls = True
        try:
            observation, _ = env.reset(seed=17)
            records = []
            for index in range(3):
                previous = observation.copy()
                observation, _, _, _, info = env.step(action)
                records.append(_record(index, previous, action, info))
            replay = replay_actions(env, records, seed=17)
            assert first_divergences(records, replay)["position_m"]["0.01"] is None
            assert first_divergences(records, replay)["heading_deg"]["0.1"] is None
            tick_sequences.append([row["executed_tick_controls"] for row in records])
        finally:
            env.close()
    assert tick_sequences[0] == tick_sequences[1]


def test_deterministic_mode_matches_public_predict_transforms() -> None:
    observations = np.zeros((4, 105), dtype=np.float32)
    for architecture, expected in (
        ("standard", np.asarray([1.0, -1.0], dtype=np.float32)),
        ("tqc_compatible", np.tanh(np.asarray([2.0, -2.0], dtype=np.float32))),
    ):
        config = TrainingConfig(
            algorithm="ppo", backend="mock", track_id="mock/straight", frame_skip=1,
            ppo=PPOConfig(
                architecture=architecture, rollout_steps=8, batch_size=8, epochs=1,
            ),
        )
        env = PolyTrackEnv(
            MockSimulatorTransport(), track_id="mock/straight", frame_skip=1,
            action_adapter=backend_for("ppo").action_adapter(config),
        )
        model = backend_for("ppo").create_model(config, env, "cpu")
        try:
            with th.no_grad():
                model.policy.action_net.weight.zero_()
                model.policy.action_net.bias.copy_(th.tensor([2.0, -2.0]))
            predicted, _ = model.predict(observations, deterministic=True)
            mode = _transformed_deterministic_mode(model, observations)
            np.testing.assert_allclose(predicted, mode, atol=1e-7, rtol=0.0)
            np.testing.assert_allclose(mode, np.broadcast_to(expected, mode.shape), atol=1e-7)
        finally:
            env.close()


def test_overlapping_air_brake_resumes_previous_layer_on_touchdown() -> None:
    class CaptureEnv(gym.Env):
        def __init__(self) -> None:
            self._air_brake_request = False
            self._air_brake_base_action = None
            self.action = None
            self.base = None

        def reset(self, *, seed=None, options=None):
            del seed, options
            observation = np.zeros(105, dtype=np.float32)
            observation[12] = 0.75
            return observation, {}

        def step(self, action):
            self.action = np.asarray(action).copy()
            self.base = self._air_brake_base_action.copy()
            return np.zeros(105, dtype=np.float32), 0.0, False, False, {}

    base = CaptureEnv()
    wrapper = AirBrakeActionWrapper(base, [
        {"kind": "air_brake", "start": 0.68, "end": 0.82, "duty": 0.02},
        {"kind": "air_brake", "start": 0.68, "end": 0.81, "duty": 1.0},
    ])
    wrapper.reset()
    wrapper.step(np.asarray([0.25, 0.8], dtype=np.float32))
    assert base._air_brake_request
    np.testing.assert_allclose(base.action, [0.25, -1.0])
    np.testing.assert_allclose(base.base, [0.25, -0.02])


def test_ppo_action_wrapper_replays_tqc_speed_schedule_and_bias_layers() -> None:
    class CaptureEnv(gym.Env):
        def __init__(self) -> None:
            self._air_brake_request = False
            self._air_brake_base_action = None

        def reset(self, *, seed=None, options=None):
            del seed, options
            observation = np.zeros(105, dtype=np.float32)
            return observation, {}

        def step(self, action):
            del action
            return np.zeros(105, dtype=np.float32), 0.0, False, False, {}

    env = CaptureEnv()
    wrapper = AirBrakeActionWrapper(
        env,
        [
            {"kind": "steer_bias", "start": 0.5, "end": 0.6, "amount": 0.02},
            {"kind": "drive_gain", "start": 0.5, "end": 0.6, "amount": 0.5},
            {"kind": "drive_bias", "start": 0.5, "end": 0.6, "amount": -0.1},
        ],
        [[0.5, 0.75, 0.1]],
    )
    observation = np.zeros(105, dtype=np.float32)
    observation[12] = 0.55
    observation[17:21] = 1.0
    np.testing.assert_allclose(
        wrapper.transform_action(np.asarray([0.2, 0.8]), observation),
        [0.22, 0.35], atol=1e-7,
    )
    wrapper._observation = observation
    _, _, _, _, info = wrapper.step(np.asarray([0.2, 0.8]))
    np.testing.assert_allclose(info["raw_policy_action"], [0.2, 0.8], atol=1e-7)
    np.testing.assert_allclose(info["transformed_policy_action"], [0.22, 0.35], atol=1e-7)


def test_tqc_residual_ppo_graft_starts_exact_and_freezes_teacher_actor(tmp_path) -> None:
    config = TrainingConfig(
        algorithm="ppo", backend="mock", track_id="mock/straight", frame_skip=1,
        timesteps=8, ppo=PPOConfig(
            architecture="tqc_residual", rollout_steps=8, batch_size=8, epochs=1,
        ),
    )
    env = PolyTrackEnv(
        MockSimulatorTransport(), track_id="mock/straight", frame_skip=1,
        action_adapter=backend_for("ppo").action_adapter(config),
    )
    model = backend_for("ppo").create_model(config, env, "cpu")
    actor = SimpleNamespace(
        features_extractor=FlattenExtractor(model.observation_space),
        latent_pi=nn.Sequential(nn.Linear(105, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()),
        mu=nn.Linear(128, 2),
    )
    teacher = SimpleNamespace(actor=actor)
    try:
        copied = initialize_actor_from_tqc(model, teacher)
        assert copied["parameters"] > 0
        observations = np.random.default_rng(12).normal(size=(64, 105)).astype(np.float32)
        with th.no_grad():
            source = th.tanh(actor.mu(actor.latent_pi(th.as_tensor(observations)))).numpy()
        student, _ = model.predict(observations, deterministic=True)
        np.testing.assert_allclose(student, source, atol=1e-7, rtol=0.0)
        assert all(
            not parameter.requires_grad
            for module in (
                model.policy.features_extractor,
                model.policy.mlp_extractor.policy_net,
                model.policy.action_net,
            )
            for parameter in module.parameters()
        )
        assert all(
            parameter.requires_grad for parameter in model.policy.residual_action.parameters()
        )
        assert all(
            th.count_nonzero(parameter) == 0
            for parameter in model.policy.residual_action.parameters()
        )
        anchor_weights = {
            name: parameter.detach().clone()
            for name, parameter in model.policy.named_parameters()
            if not parameter.requires_grad
        }

        model.learn(total_timesteps=8)
        for name, parameter in model.policy.named_parameters():
            if name in anchor_weights:
                th.testing.assert_close(parameter, anchor_weights[name], rtol=0.0, atol=0.0)
        with th.no_grad():
            model.policy.residual_action.bias.copy_(th.tensor([50.0, -50.0]))
        bounded, _ = model.predict(observations, deterministic=True)
        assert np.max(np.abs(bounded - source)) <= config.ppo.residual_action_limit + 1e-6
        sampled, _ = model.predict(observations, deterministic=False)
        assert np.all(np.isfinite(sampled))
        assert np.all(sampled >= -1.0) and np.all(sampled <= 1.0)
        checkpoint = tmp_path / "policy.zip"
        model.save(str(checkpoint))
        loaded = backend_for("ppo").load_model(checkpoint, None, "cpu")
        restored, _ = loaded.predict(observations, deterministic=True)
        np.testing.assert_allclose(restored, bounded, atol=1e-7, rtol=0.0)
        assert all(
            not parameter.requires_grad
            for module in (
                loaded.policy.features_extractor,
                loaded.policy.mlp_extractor.policy_net,
                loaded.policy.action_net,
            )
            for parameter in module.parameters()
        )
        assert all(
            parameter.requires_grad for parameter in loaded.policy.residual_action.parameters()
        )
        assert np.all(np.isfinite(restored))
        assert np.all(restored >= -1.0) and np.all(restored <= 1.0)
    finally:
        env.close()


def test_tqc_compatible_graft_supports_longer_rollouts_and_updates_full_actor() -> None:
    config = TrainingConfig(
        algorithm="ppo", backend="mock", track_id="mock/straight", frame_skip=1,
        timesteps=8, ppo=PPOConfig(
            architecture="tqc_compatible", rollout_steps=4096, batch_size=256, epochs=1,
        ),
    )
    env = PolyTrackEnv(
        MockSimulatorTransport(), track_id="mock/straight", frame_skip=1,
        action_adapter=backend_for("ppo").action_adapter(config),
    )
    model = backend_for("ppo").create_model(config, env, "cpu")
    actor = SimpleNamespace(
        features_extractor=FlattenExtractor(model.observation_space),
        latent_pi=nn.Sequential(nn.Linear(105, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()),
        mu=nn.Linear(128, 2),
    )
    teacher = SimpleNamespace(actor=actor)
    try:
        copied = initialize_actor_from_tqc(model, teacher)
        observations = np.random.default_rng(18).normal(size=(32, 105)).astype(np.float32)
        with th.no_grad():
            expected = th.tanh(actor.mu(actor.latent_pi(th.as_tensor(observations)))).numpy()
        actual, _ = model.predict(observations, deterministic=True)
        assert copied["parameters"] > 0
        assert model.n_steps == 4096
        assert model.batch_size == 256
        assert not hasattr(model.policy, "residual_action")
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=0.0)
        assert all(
            parameter.requires_grad
            for module in (
                model.policy.features_extractor,
                model.policy.mlp_extractor.policy_net,
                model.policy.action_net,
            )
            for parameter in module.parameters()
        )
    finally:
        env.close()


def test_tqc_residual_can_be_gated_to_a_progress_window() -> None:
    config = TrainingConfig(
        algorithm="ppo", backend="mock", track_id="mock/straight", frame_skip=1,
        timesteps=8, ppo=PPOConfig(
            architecture="tqc_residual", rollout_steps=8, batch_size=8, epochs=1,
            residual_progress_start=0.4, residual_progress_end=0.6,
        ),
    )
    env = PolyTrackEnv(
        MockSimulatorTransport(), track_id="mock/straight", frame_skip=1,
        action_adapter=backend_for("ppo").action_adapter(config),
    )
    model = backend_for("ppo").create_model(config, env, "cpu")
    try:
        with th.no_grad():
            model.policy.residual_action.bias.copy_(th.tensor([50.0, -50.0]))
            observations = th.zeros((2, 105))
            observations[0, 12] = 0.3
            observations[1, 12] = 0.5
            features = model.policy.extract_features(observations)
            latent = model.policy.mlp_extractor.forward_actor(features)
            base = th.tanh(model.policy.action_net(latent))
        actions, _ = model.predict(observations.numpy(), deterministic=True)
        np.testing.assert_allclose(actions[0], base[0].numpy(), atol=1e-7, rtol=0.0)
        assert np.max(np.abs(actions[1] - base[1].numpy())) > 0.05
    finally:
        env.close()
