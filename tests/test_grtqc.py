from __future__ import annotations

import json
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th
from sb3_contrib import TQC
from stable_baselines3.common.buffers import NStepReplayBuffer
from stable_baselines3.common.logger import configure

from polybot.algorithms.grtqc import GRTQCBackend
from polybot.algorithms.tqc import TQCBackend
from polybot.environment.env import PolyTrackEnv
from polybot.mock import MockSimulatorTransport
from polybot.models.registry import IncompatibleModelError, ModelMetadata, ModelRegistry
from polybot.protocol import ProtocolViolation
from polybot.training.config import EvaluationConfig, GRTQCConfig, TQCConfig, TrainingConfig
from polybot.training.evaluation import EvaluationResult, PrefixObservationReference
from polybot.training.runner import TrainingRunner
from tools.audit_grtqc_training_distribution import TrainingDistributionDriver


def _config(algorithm: str) -> TrainingConfig:
    settings = {
        "tqc": {"tqc": TQCConfig(architecture="tiny", learning_starts=8, batch_size=8)},
        "grtqc": {"grtqc": GRTQCConfig(
            architecture="tiny", learning_starts=8, batch_size=8,
            train_frequency=1, critic_warmup_updates=100,
            critic_readiness_window=8,
        )},
    }
    return TrainingConfig(
        algorithm=algorithm, evaluation=EvaluationConfig(16, 5),
        **settings[algorithm],
    )


def _model(config: TrainingConfig):
    backend = TQCBackend() if config.algorithm == "tqc" else GRTQCBackend()
    env = PolyTrackEnv(
        MockSimulatorTransport(), action_adapter=backend.action_adapter(config),
        expose_training_state=bool(config.grtqc and config.grtqc.critic_environment_state),
    )
    return backend.create_model(config, env, "cpu"), env


def test_gated_actor_transfers_all_tqc_weights_with_near_identical_actions() -> None:
    source, source_env = _model(_config("tqc"))
    target, target_env = _model(_config("grtqc"))
    try:
        transfer = target.actor.load_state_dict(source.actor.state_dict(), strict=False)
        assert not transfer.unexpected_keys
        assert len(transfer.missing_keys) == 4
        assert all(".gate." in key for key in transfer.missing_keys)
        observations = np.random.default_rng(7).normal(
            size=(64, source.observation_space.shape[0]),
        ).astype(np.float32)
        with th.no_grad():
            inputs = th.as_tensor(observations)
            left = source.actor(inputs, deterministic=True)
            right = target.actor(inputs, deterministic=True)
        th.testing.assert_close(left, right, rtol=0, atol=0)
    finally:
        source_env.close()
        target_env.close()


def test_grtqc_without_extra_penalty_matches_the_installed_tqc_update_mathematically():
    config = _config("grtqc")
    config.grtqc.disagreement_coefficient = 0
    config.grtqc.actor_step_action_limit = 2
    model, env = _model(config)
    reference, reference_env = _model(config)
    try:
        rng = np.random.default_rng(19)
        for _ in range(32):
            observation = rng.normal(size=(1, 105)).astype(np.float32)
            following = rng.normal(size=(1, 105)).astype(np.float32)
            action = rng.uniform(-1, 1, size=(1, 2)).astype(np.float32)
            reward = rng.normal(size=1).astype(np.float32)
            done = rng.random(size=1) > .7
            for learner in (model, reference):
                learner.replay_buffer.add(observation, following, action, reward, done, [{}])
        reference.policy.load_state_dict(model.policy.state_dict())
        for learner in (model, reference):
            learner.actor_unlocked = True
            learner.set_logger(configure(None, []))
            np.random.seed(17)
            th.manual_seed(17)
            if learner is model:
                learner.train(3, 8)
            else:
                TQC.train(learner, 3, 8)
        for name, tensor in reference.policy.state_dict().items():
            th.testing.assert_close(tensor, model.policy.state_dict()[name], rtol=1e-6, atol=2e-7)
        th.testing.assert_close(model.log_ent_coef, reference.log_ent_coef, rtol=0, atol=0)
    finally:
        env.close()
        reference_env.close()


def test_critic_environment_context_does_not_change_actor_inputs_or_transfer(tmp_path):
    source, source_env = _model(_config("tqc"))
    config = _config("grtqc")
    config.grtqc.critic_controller_state = True
    config.grtqc.actor_controller_state = True
    config.grtqc.critic_environment_state = True
    target, target_env = _model(config)
    try:
        target.actor.load_state_dict(source.actor.state_dict(), strict=False)
        base = np.random.default_rng(17).normal(size=(64, 105)).astype(np.float32)
        extra = np.random.default_rng(18).uniform(-1, 1, size=(64, 16)).astype(np.float32)
        augmented = np.concatenate((base, extra), axis=1)
        assert target.actor.features_extractor.features_dim == 109
        assert target.critic.features_extractor.features_dim == 121
        np.testing.assert_array_equal(
            source.predict(base, deterministic=True)[0], target.predict(augmented, deterministic=True)[0],
        )
        changed = augmented.copy()
        changed[:, 109:] += 1
        np.testing.assert_array_equal(
            target.predict(augmented, deterministic=True)[0], target.predict(changed, deterministic=True)[0],
        )
        backend = GRTQCBackend()
        backend.save_model(target, tmp_path)
        loaded = backend.load_model(tmp_path / "policy.zip", None, "cpu")
        np.testing.assert_array_equal(
            target.predict(augmented, deterministic=True)[0], loaded.predict(augmented, deterministic=True)[0],
        )
        old, old_env = _model(_config("grtqc"))
        try:
            with pytest.raises(ValueError, match="observation layout"):
                backend.configure_resume(old, config, "cpu")
        finally:
            old_env.close()
    finally:
        source_env.close()
        target_env.close()


def test_incomplete_deadline_masks_bootstrap_in_replay():
    config = _config("grtqc")
    model, env = _model(config)
    try:
        env.max_episode_steps = 1
        observation = model.env.reset()
        action = np.array([[0, 1]], dtype=np.float32)
        _, rewards, dones, infos = model.env.step(action)
        assert dones[0] and not infos[0]["TimeLimit.truncated"]
        terminal = infos[0]["terminal_observation"][None]
        model.replay_buffer.add(observation, terminal, action, rewards, dones, infos)
        assert float(model.replay_buffer.sample(1).dones[0]) == 1
    finally:
        env.close()


def test_controller_adapter_settings_require_real_controller_observations():
    with pytest.raises(ValueError, match="controller-state observations"):
        GRTQCConfig(actor_controller_state=True)
    with pytest.raises(ValueError, match="actor controller inputs"):
        GRTQCConfig(controller_adapter_only=True)


def test_critic_updates_keep_transferred_actor_frozen_until_ready() -> None:
    model, env = _model(_config("grtqc"))
    try:
        before = {name: weight.clone() for name, weight in model.actor.state_dict().items()}
        model.learn(24)
        assert model.critic_updates_since_transfer > 0
        assert not model.actor_unlocked
        for name, weight in model.actor.state_dict().items():
            th.testing.assert_close(weight, before[name])
        metrics = GRTQCBackend().metrics(model)
        assert metrics["critic_disagreement"] is not None
        assert metrics["disagreement_penalty"] is not None
    finally:
        env.close()


def test_complete_return_initialization_changes_only_critics_and_survives_reload(tmp_path):
    config = _config("grtqc")
    config.grtqc.critic_mc_initialization_updates = 4
    config.grtqc.critic_mc_min_episodes = 1
    model, env = _model(config)
    try:
        model.set_logger(configure(None, []))
        observation = np.zeros((1, model.observation_space.shape[0]), dtype=np.float32)
        action, _ = model.predict(observation, deterministic=True)
        for _ in range(2):
            model.replay_buffer.add(observation, observation, action, np.ones(1), np.ones(1), [{}])
        before = {name: value.clone() for name, value in model.actor.state_dict().items()}
        model.train(10, 4)
        assert model.critic_mc_updates_done == 4
        assert model.critic_updates_since_transfer == 14
        assert model._n_updates == 14
        assert model._critic_reference_episodes == 2
        assert not model.actor_unlocked
        for name, value in model.actor.state_dict().items():
            th.testing.assert_close(value, before[name], rtol=0, atol=0)
        backend = GRTQCBackend()
        backend.save_model(model, tmp_path, resume=True)
        loaded = backend.load_model(tmp_path / "policy.zip", None, "cpu", resume=True)
        loaded.set_logger(configure(None, []))
        assert loaded.critic_mc_updates_done == 4
        assert loaded._critic_reference_observations is None
        loaded.train(1, 4)
        assert loaded.critic_mc_updates_done == 4
        assert loaded._n_updates == 15
        assert loaded._critic_reference_episodes == 2
        backend.configure_resume(loaded, config, "cpu", fresh_replay=True)
        assert loaded.critic_mc_updates_done == 0
        assert loaded._critic_reference_observations is None
    finally:
        env.close()


def test_return_initialization_blocks_unlock_until_policy_values_are_accurate():
    config = _config("grtqc")
    config.grtqc.critic_mc_initialization_updates = 1
    model, env = _model(config)
    try:
        model.critic_updates_since_transfer = model.critic_warmup_updates
        model._critic_loss_history.extend([1.0] * model.critic_readiness_window)
        model._disagreement_history.extend([0.1] * model.critic_readiness_window)
        assert not model._critic_ready()
        model.critic_mc_updates_done = 1
        model._critic_reference_observations = th.zeros((4, model.observation_space.shape[0]))
        model._critic_reference_actions = th.zeros((4, 2))
        model._critic_reference_returns = th.full((4, 1), 100.0)
        assert not model._critic_ready()
        assert model._critic_reference_error > config.grtqc.critic_reference_error_limit
        model._critic_reference_error = 0.01
        model._critic_reference_probe_update = model.critic_updates_since_transfer
        assert model._critic_ready()
    finally:
        env.close()


def test_resume_repairs_replay_sampling_class_even_when_model_horizon_already_matches():
    config = _config("grtqc")
    config.grtqc.n_step_return = 4
    model, env = _model(config)
    try:
        replay = model.replay_buffer
        assert type(replay) is NStepReplayBuffer
        model.n_steps = 1  # Policy snapshot coupled with another checkpoint's raw replay.
        assert model.configure_replay_horizon(1, config.grtqc.gamma)
        assert type(model.replay_buffer) is not NStepReplayBuffer
        assert model.replay_buffer.observations is replay.observations
        assert not model.configure_replay_horizon(1, config.grtqc.gamma)
    finally:
        env.close()


def test_controller_critics_keep_exact_actor_transfer_and_survive_reload(tmp_path) -> None:
    source, source_env = _model(_config("tqc"))
    config = _config("grtqc")
    config.grtqc.critic_controller_state = True
    target, target_env = _model(config)
    try:
        transferred = target.actor.load_state_dict(source.actor.state_dict(), strict=False)
        assert not transferred.unexpected_keys
        assert len(transferred.missing_keys) == 4
        base = np.random.default_rng(17).normal(size=(64, 105)).astype(np.float32)
        controller = np.random.default_rng(18).uniform(-1, 1, size=(64, 4)).astype(np.float32)
        augmented = np.concatenate((base, controller), axis=1)
        expected = source.predict(base, deterministic=True)[0]
        np.testing.assert_array_equal(expected, target.predict(augmented, deterministic=True)[0])
        altered = augmented.copy()
        altered[:, -4:] = 0
        np.testing.assert_array_equal(expected, target.predict(altered, deterministic=True)[0])
        with th.no_grad():
            actions = th.as_tensor(expected)
            q_a = target.critic(th.as_tensor(augmented), actions)
            q_b = target.critic(th.as_tensor(altered), actions)
            assert (q_a - q_b).abs().max() > 0
        reference = PrefixObservationReference(source, extra_features=4)
        np.testing.assert_array_equal(expected, reference.predict(augmented, deterministic=True)[0])
        with pytest.raises(ValueError, match="declared controller-state"):
            reference.predict(base, deterministic=True)
        GRTQCBackend().save_model(target, tmp_path)
        loaded = GRTQCBackend().load_model(tmp_path / "policy.zip", None, "cpu")
        np.testing.assert_array_equal(expected, loaded.predict(augmented, deterministic=True)[0])
        target.learn(24)
        assert target.replay_buffer.observations.shape[-1] == 109
        assert target.critic_updates_since_transfer > 0
    finally:
        source_env.close()
        target_env.close()


def test_pending_evaluation_holds_actor_but_keeps_real_critic_updates() -> None:
    config = _config("grtqc")
    model, env = _model(config)
    try:
        model.actor_unlocked = True
        model._actor_evaluation_hold = True
        before = {name: weight.clone() for name, weight in model.actor.state_dict().items()}
        model.learn(24)
        assert model.critic_updates_since_transfer > 0
        assert model.replay_buffer.size() == 24
        assert model.actor_unlocked
        for name, weight in model.actor.state_dict().items():
            th.testing.assert_close(weight, before[name], rtol=0, atol=0)
        assert GRTQCBackend().metrics(model)["actor_evaluation_pending"] == 1
        GRTQCBackend().configure_resume(model, config, "cpu")
        assert not model._actor_evaluation_hold
    finally:
        env.close()


def test_controller_actor_adapter_preserves_source_then_learns_without_changing_inherited_weights(tmp_path):
    source, source_env = _model(_config("tqc"))
    config = _config("grtqc")
    config.grtqc.critic_controller_state = True
    config.grtqc.actor_controller_state = True
    config.grtqc.controller_adapter_only = True
    target, target_env = _model(config)
    try:
        result = target.actor.load_state_dict(source.actor.state_dict(), strict=False)
        assert len(result.missing_keys) == 5
        assert "latent_pi.0.controller_weight" in result.missing_keys
        base = np.random.default_rng(17).normal(size=(64, 105)).astype(np.float32)
        controller = np.random.default_rng(18).uniform(-1, 1, size=(64, 4)).astype(np.float32)
        augmented = np.concatenate((base, controller), axis=1)
        expected = source.predict(base, deterministic=True)[0]
        np.testing.assert_array_equal(expected, target.predict(augmented, deterministic=True)[0])
        before = {name: weight.clone() for name, weight in target.actor.state_dict().items()}
        target.actor_unlocked = True
        target.learn(24)
        changed = []
        for name, weight in target.actor.state_dict().items():
            if not th.equal(weight, before[name]):
                changed.append(name)
        assert changed == ["latent_pi.0.controller_weight"]
        assert target.critic_updates_since_transfer > 0
        action = target.predict(augmented, deterministic=True)[0]
        backend = GRTQCBackend()
        counts = backend.parameter_counts(target)
        assert counts["actor_trainable"] == 4 * target.actor.latent_pi[0].out_features
        assert counts["actor"] > counts["actor_trainable"]
        backend.save_model(target, tmp_path, resume=True)
        loaded = backend.load_model(tmp_path / "policy.zip", None, "cpu", resume=True)
        np.testing.assert_array_equal(action, loaded.predict(augmented, deterministic=True)[0])
        assert loaded.policy.controller_adapter_only
        assert sum(p.requires_grad for p in loaded.actor.parameters()) == 1
        loaded.set_env(target_env)
        loaded.learn(8, reset_num_timesteps=False)
        assert loaded.actor.optimizer.state
        config.grtqc.controller_adapter_only = False
        backend.configure_resume(loaded, config, "cpu")
        assert all(p.requires_grad for p in loaded.actor.parameters())
        before_full_learning = {name: value.clone() for name, value in loaded.actor.state_dict().items()}
        loaded.learn(8, reset_num_timesteps=False)
        assert any(
            not th.equal(value, before_full_learning[name])
            for name, value in loaded.actor.state_dict().items()
            if name != "latent_pi.0.controller_weight"
        )
    finally:
        source_env.close()
        target_env.close()


def test_controller_adapter_upgrade_retains_replay_critics_and_exact_fallback(tmp_path):
    config = _config("grtqc")
    config.grtqc.critic_controller_state = True
    model, env = _model(config)
    fallback, fallback_env = _model(config)
    try:
        model.learn(24)
        observations = model.replay_buffer.observations[:24, 0].copy()
        expected = model.predict(observations, deterministic=True)[0]
        replay, critic_optimizer = model.replay_buffer, model.critic.optimizer
        critic = {name: value.clone() for name, value in model.critic.state_dict().items()}
        config.grtqc.actor_controller_state = True
        config.grtqc.controller_adapter_only = True
        backend = GRTQCBackend()
        backend.configure_resume(model, config, "cpu")
        assert model.replay_buffer is replay
        assert model.critic.optimizer is critic_optimizer
        assert not model.actor_unlocked
        assert model.critic_updates_since_transfer == 0
        for name, value in model.critic.state_dict().items():
            th.testing.assert_close(value, critic[name], rtol=0, atol=0)
        np.testing.assert_array_equal(expected, model.predict(observations, deterministic=True)[0])
        backend.save_model(model, tmp_path, resume=True)
        loaded = backend.load_model(tmp_path / "policy.zip", None, "cpu", resume=True)
        np.testing.assert_array_equal(expected, loaded.predict(observations, deterministic=True)[0])
        with th.no_grad():
            model.actor.latent_pi[0].controller_weight.fill_(0.1)
        backend.restore_actor_weights(model, fallback)
        assert th.count_nonzero(model.actor.latent_pi[0].controller_weight) == 0
        np.testing.assert_array_equal(
            fallback.predict(observations, deterministic=True)[0],
            model.predict(observations, deterministic=True)[0],
        )
        config.grtqc.actor_controller_state = False
        config.grtqc.controller_adapter_only = False
        with pytest.raises(ValueError, match="cannot disable"):
            backend.configure_resume(model, config, "cpu")
    finally:
        env.close()
        fallback_env.close()


@pytest.mark.parametrize("setting,value", [("policy_std_limit", 0.001), ("gamma", 0.999)])
def test_changed_critic_targets_repeat_warmup_with_saved_replay(setting, value):
    config = _config("grtqc")
    model, env = _model(config)
    try:
        model.actor_unlocked = True
        model.critic_updates_since_transfer = 9000
        replay = model.replay_buffer
        setattr(config.grtqc, setting, value)
        GRTQCBackend().configure_resume(model, config, "cpu")
        assert not model.actor_unlocked
        assert model.critic_updates_since_transfer == 0
        assert model.replay_buffer is replay
        model.actor_unlocked = True
        model.critic_updates_since_transfer = 9000
        GRTQCBackend().configure_resume(model, config, "cpu")
        assert model.actor_unlocked
        assert model.critic_updates_since_transfer == 9000
    finally:
        env.close()


def test_controller_state_layout_cannot_silently_reuse_legacy_model_or_replay(tmp_path) -> None:
    legacy = _config("grtqc")
    metadata = ModelMetadata(
        algorithm="grtqc", architecture="tiny", actor_parameters=0, critic_parameters=0,
        total_trainable_parameters=0, observation_schema="polybot.observation.v2",
        action_schema="continuous-pwm-v2", track_name=legacy.track_name, track_id=legacy.track_id,
        lookahead_count=legacy.lookahead_count, reward_profile=None, curriculum={},
        training_config=legacy.to_dict(), training_timesteps=0, simulator_ticks=0, wall_seconds=0,
        seed=0, device="cpu", finishes=0, crashes=0,
    )
    registry = ModelRegistry(tmp_path)
    registry.validate(metadata, legacy, "continuous-pwm-v2")
    augmented = _config("grtqc")
    augmented.grtqc.critic_controller_state = True
    with pytest.raises(IncompatibleModelError, match="observation"):
        registry.validate(metadata, augmented, "continuous-pwm-v2")
    metadata.observation_schema = "polybot.observation.v2.pwm-state"
    registry.validate(metadata, augmented, "continuous-pwm-v2")
    with pytest.raises(IncompatibleModelError, match="observation"):
        registry.validate(metadata, legacy, "continuous-pwm-v2")


def test_pending_evaluation_waits_for_episode_and_allows_phase_boundary(tmp_path, monkeypatch) -> None:
    config = replace(_config("grtqc"), output_root=tmp_path / "models")
    config.grtqc.finish_episode_before_actor_eval = True
    runner = TrainingRunner(config)
    runner.model = SimpleNamespace(num_timesteps=512, actor_unlocked=True, _actor_evaluation_hold=False)
    events = []
    monkeypatch.setattr(runner, "_emit", events.append)
    assert runner._defer_actor_evaluation(phase_finished=False)
    assert runner.model._actor_evaluation_hold
    assert runner._defer_actor_evaluation(phase_finished=False)
    assert len(events) == 1
    runner._actor_eval_episode_finished = True
    assert not runner._defer_actor_evaluation(phase_finished=False)
    runner.model._actor_evaluation_hold = False
    assert not runner._defer_actor_evaluation(phase_finished=True)
    runner.model.actor_unlocked = False
    assert not runner._defer_actor_evaluation(phase_finished=False)


def test_actor_step_backtracks_large_action_change() -> None:
    config = _config("grtqc")
    assert config.grtqc is not None
    config.grtqc.actor_learning_rate = 1e-3
    config.grtqc.actor_step_action_limit = 1e-5
    model, env = _model(config)
    try:
        reference_observations = np.random.default_rng(11).normal(
            size=(256, model.observation_space.shape[0]),
        ).astype(np.float32)
        model.set_actor_reference_observations(reference_observations)
        model.actor_unlocked = True
        model.learn(16)
        metrics = GRTQCBackend().metrics(model)
        assert metrics["actor_proposed_action_drift"] is not None
        assert metrics["actor_executed_action_drift"] is not None
        assert metrics["actor_proposed_action_drift"] > config.grtqc.actor_step_action_limit
        assert metrics["actor_executed_action_drift"] <= 1.1e-5
        assert metrics["actor_reference_action_drift"] <= 1.1e-5
        assert metrics["actor_reference_cumulative_action_drift"] <= (
            config.grtqc.actor_reference_drift_limit
        )
    finally:
        env.close()


def test_disabled_reference_cap_allows_learning_with_bounded_steps() -> None:
    config = _config("grtqc")
    config.grtqc.actor_learning_rate = 1e-3
    config.grtqc.actor_step_action_limit = 1e-4
    config.grtqc.actor_reference_drift_limit = 0.0
    model, env = _model(config)
    try:
        observations = np.random.default_rng(11).normal(
            size=(256, model.observation_space.shape[0]),
        ).astype(np.float32)
        model.set_actor_reference_observations(observations)
        model.actor_unlocked = True
        model.learn(24)
        metrics = GRTQCBackend().metrics(model)
        assert metrics["actor_reference_cumulative_action_drift"] > 0
        assert 0 < metrics["actor_executed_action_drift"] <= 1.01e-4
        assert metrics["actor_reference_action_drift"] <= 1.01e-4
    finally:
        env.close()


def test_correlated_exploration_resets_at_episode_end(monkeypatch) -> None:
    config = _config("grtqc")
    config.grtqc.exploration_correlation = 0.9
    model, env = _model(config)
    try:
        rng = np.random.default_rng(config.seed)
        first = model._rollout_noise((1, 2), 0.006)
        expected_first = np.sqrt(1 - 0.9**2) * rng.normal(0, 0.006, (1, 2))
        np.testing.assert_allclose(first, expected_first, rtol=1e-6)
        second = model._rollout_noise((1, 2), 0.006)
        expected_second = 0.9 * first + np.sqrt(1 - 0.9**2) * rng.normal(0, 0.006, (1, 2))
        np.testing.assert_allclose(second, expected_second, rtol=1e-6)
        monkeypatch.setattr("polybot.algorithms.tqc.SeededWarmupTQC._store_transition", lambda *a: None)
        model._store_transition(None, first, None, np.zeros(1), np.array([True]), [{}])
        np.testing.assert_array_equal(model._exploration_noise, np.zeros((1, 2)))
        model._rollout_noise((1, 2), 0.006)
        np.testing.assert_array_equal(model._rollout_noise((1, 2), 0), np.zeros((1, 2)))
    finally:
        env.close()


def test_bounded_training_distribution_preserves_mean_and_variance_gradients() -> None:
    config = _config("grtqc")
    config.grtqc.policy_std_limit = 0.02
    model, env = _model(config)
    try:
        model.learn(8)
        observations = th.zeros((16, model.observation_space.shape[0]))
        before = model.actor(observations, deterministic=True).detach().clone()
        actions, log_probability = model._training_actions_log_prob(observations)
        distribution = model.actor.action_dist.distribution
        assert float(distribution.stddev.detach().max()) <= 0.02
        expected_probability = model.actor.action_dist.log_prob(actions)
        th.testing.assert_close(log_probability, expected_probability)
        th.testing.assert_close(model.actor(observations, deterministic=True), before, rtol=0, atol=0)
        model.actor.optimizer.zero_grad()
        (actions.mean() + log_probability.mean()).backward()
        variance_gradient = model.actor.log_std.bias.grad
        assert th.isfinite(variance_gradient).all()
        assert th.count_nonzero(variance_gradient) > 0
    finally:
        env.close()


def test_rollout_passes_air_brake_touchdown_release_to_wrapped_environment() -> None:
    config = _config("grtqc")
    config.grtqc.critic_collection_std = 0.0
    model, env = _model(config)
    try:
        model.policy_overlays = [{"kind": "air_brake", "start": 0.2, "end": 0.8, "duty": 0.05}]
        observation = model.env.reset()
        observation[0, 12] = 0.5
        observation[0, 17:21] = 0
        model._last_obs = observation
        action, _ = model._sample_action(0)
        assert action[0, 1] == pytest.approx(-0.05)
        assert env._air_brake_request
        np.testing.assert_array_equal(env._air_brake_base_action, model._air_brake_base_action[0])
    finally:
        env.close()


def test_distribution_audit_samples_before_overlays_and_restores_normal_prediction(monkeypatch):
    model, env = _model(_config("grtqc"))
    try:
        observations = model.env.reset()
        observations[0, 12] = 0.5
        observations[0, 17:21] = 0
        model.policy_overlays = [{"kind": "air_brake", "start": 0.2, "end": 0.8, "duty": 0.5}]
        original = model.policy.predict
        before = {name: value.clone() for name, value in model.actor.state_dict().items()}
        monkeypatch.setattr(model, "_training_actions_log_prob", lambda tensor: (
            tensor.new_tensor([[0.125, 0.75]]), tensor.new_tensor([3.4]),
        ))
        driver = TrainingDistributionDriver(model)
        action, _ = driver.predict(observations[0], deterministic=True)
        np.testing.assert_array_equal(action, [0.125, -0.5])
        np.testing.assert_array_equal(driver._air_brake_base_action, [0.125, 0.75])
        assert driver._air_brake_active
        assert model.policy.predict == original
        assert driver.log_probabilities == pytest.approx([3.4])
        for name, value in model.actor.state_dict().items():
            th.testing.assert_close(value, before[name], rtol=0, atol=0)
    finally:
        env.close()


def test_curriculum_boundary_keeps_full_lap_bootstrap_in_replay() -> None:
    config = _config("grtqc")
    backend = GRTQCBackend()
    env = PolyTrackEnv(
        MockSimulatorTransport(), track_id="mock/straight", frame_skip=4,
        curriculum_start_ratio=0, curriculum_end_ratio=0.01,
        action_adapter=backend.action_adapter(config),
    )
    model = backend.create_model(config, env, "cpu")
    try:
        observations = model.env.reset()
        action = np.array([[0, 1]], dtype=np.float32)
        for _ in range(200):
            next_observations, rewards, dones, infos = model.env.step(action)
            if dones[0]:
                assert "curriculum_section_complete" in infos[0]["events"]
                assert infos[0]["TimeLimit.truncated"] is True
                terminal = infos[0]["terminal_observation"][None]
                model.replay_buffer.add(observations, terminal, action, rewards, dones, infos)
                sample = model.replay_buffer.sample(1)
                assert float(sample.dones[0]) == 0.0
                th.testing.assert_close(sample.next_observations[0], th.as_tensor(terminal[0]))
                break
            observations = next_observations
        else:
            pytest.fail("mock driver never reached the curriculum boundary")
    finally:
        env.close()


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_critic_exploration_preserves_initial_reliable_fill_and_quiet_episodes(fraction) -> None:
    config = _config("grtqc")
    config.grtqc.critic_collection_std = 0.005
    config.grtqc.critic_exploration_fraction = fraction
    model, env = _model(config)
    try:
        model._last_obs = model.env.reset()
        expected, _ = model.predict(model._last_obs, deterministic=True)
        initial, _ = model._sample_action(config.grtqc.learning_starts)
        np.testing.assert_array_equal(initial, expected)
        model.num_timesteps = config.grtqc.learning_starts
        collected, _ = model._sample_action(config.grtqc.learning_starts)
        if fraction == 0:
            np.testing.assert_array_equal(collected, expected)
        else:
            assert not np.array_equal(collected, expected)
    finally:
        env.close()


def test_explicit_entropy_change_applies_on_resume_without_resetting_unchanged_runs() -> None:
    config = _config("grtqc")
    model, env = _model(config)
    try:
        with th.no_grad():
            model.log_ent_coef.fill_(log_value := -4.0)
        backend = GRTQCBackend()
        backend.configure_resume(model, config, "cpu")
        assert float(model.log_ent_coef.detach()) == log_value
        config.grtqc.entropy = "auto_0.0001"
        backend.configure_resume(model, config, "cpu")
        assert float(model.log_ent_coef.detach().exp()) == pytest.approx(0.0001)
    finally:
        env.close()


def test_multistep_rewards_stop_at_finish_and_at_evaluation_reset() -> None:
    config = _config("grtqc")
    config.grtqc.n_step_return = 4
    config.grtqc.gamma = 0.9
    model, env = _model(config)
    try:
        replay = model.replay_buffer
        assert isinstance(replay, NStepReplayBuffer)
        shape = (1, model.observation_space.shape[0])
        action = np.zeros((1, 2), dtype=np.float32)
        # Two observed decisions before an evaluation interrupts the episode.
        for index in range(2):
            replay.add(np.full(shape, index), np.full(shape, index + 1), action,
                       np.array([1.0]), np.array([False]), [{}])
        model.set_env(env)
        # This is a different episode with a much larger finish reward.
        replay.add(np.full(shape, 10), np.full(shape, 11), action,
                   np.array([50.0]), np.array([True]), [{}])
        sample = replay._get_samples(np.array([0, 2]))
        th.testing.assert_close(sample.rewards[:, 0], th.tensor([1.9, 50.0]))
        th.testing.assert_close(sample.dones[:, 0], th.tensor([0.0, 1.0]))
        th.testing.assert_close(sample.discounts[:, 0], th.tensor([0.81, 0.9]))
        th.testing.assert_close(sample.next_observations[0], th.full(shape[1:], 2.0))
    finally:
        env.close()


def test_resume_changes_horizon_without_losing_replay_or_actor_and_repairs_old_resets() -> None:
    config = _config("grtqc")
    model, env = _model(config)
    try:
        replay = model.replay_buffer
        shape = (1, model.observation_space.shape[0])
        action = np.zeros((1, 2), dtype=np.float32)
        replay.add(np.zeros(shape), np.ones(shape), action, np.array([1.0]), np.array([False]), [{}])
        replay.add(np.full(shape, 10), np.full(shape, 11), action,
                   np.array([50.0]), np.array([True]), [{}])
        actor = {name: weight.clone() for name, weight in model.actor.state_dict().items()}
        model.actor_unlocked = True
        model.critic_updates_since_transfer = 3000
        config.grtqc.n_step_return = 4
        config.grtqc.gamma = 0.9
        GRTQCBackend().configure_resume(model, config, "cpu")
        assert isinstance(model.replay_buffer, NStepReplayBuffer)
        assert model.replay_buffer.observations is replay.observations
        assert model.replay_buffer.size() == 2
        assert not model.actor_unlocked
        assert model.critic_updates_since_transfer == 0
        sample = model.replay_buffer._get_samples(np.array([0]))
        assert float(sample.rewards[0]) == 1.0
        assert float(sample.dones[0]) == 0.0
        assert float(sample.discounts[0]) == pytest.approx(0.9)
        for name, weight in model.actor.state_dict().items():
            th.testing.assert_close(weight, actor[name], rtol=0, atol=0)
        config.grtqc.gamma = 0.8
        GRTQCBackend().configure_resume(model, config, "cpu")
        assert model.replay_buffer.gamma == 0.8
    finally:
        env.close()


def test_fresh_replay_resume_refreezes_actor_and_collects_reliable_initial_laps() -> None:
    config = _config("grtqc")
    model, env = _model(config)
    try:
        model.actor_unlocked = True
        model.critic_updates_since_transfer = 3000
        model._critic_loss_history.append(1.0)
        model._disagreement_history.append(0.1)
        model.num_timesteps = 5000
        actor = {name: weight.clone() for name, weight in model.actor.state_dict().items()}
        GRTQCBackend().configure_resume(model, config, "cpu", fresh_replay=True)
        assert not model.actor_unlocked
        assert model.critic_updates_since_transfer == 0
        assert not model._critic_loss_history
        assert not model._disagreement_history
        assert model.learning_starts > model.num_timesteps
        model._last_obs = model.env.reset()
        expected, _ = model.predict(model._last_obs, deterministic=True)
        action, _ = model._sample_action(model.learning_starts)
        np.testing.assert_array_equal(action, expected)
        for name, weight in model.actor.state_dict().items():
            th.testing.assert_close(weight, actor[name], rtol=0, atol=0)
    finally:
        env.close()


@pytest.mark.parametrize("algorithm", ["tqc", "grtqc"])
def test_old_reward_semantics_cannot_be_reused_from_replay(tmp_path, monkeypatch, algorithm) -> None:
    config = replace(_config(algorithm), output_root=tmp_path / "models", log_root=tmp_path / "logs")
    runner = TrainingRunner(config)
    monkeypatch.setattr(runner.registry, "validate", lambda *args: None)
    monkeypatch.setattr(runner.registry, "read_metadata", lambda *args: SimpleNamespace(
        training_config=config.to_dict(), reward_semantics="executed-controls-v1",
        critic_adaptation_required=False, architecture="tiny",
    ))
    with pytest.raises(ValueError, match="resume reward settings differ"):
        runner.run(resume=tmp_path / "latest")


@pytest.mark.parametrize(
    ("laps", "expected"),
    [((24.2,) * 5, "champion"), ((24.3,) * 5, "rejected"), ((24.1,) * 4, "rejected")],
)
def test_promotion_requires_five_reliable_faster_laps(tmp_path, monkeypatch, laps, expected) -> None:
    config = replace(_config("grtqc"), output_root=tmp_path / "models")
    runner = TrainingRunner(config)
    runner.model = SimpleNamespace(num_timesteps=5000)
    result = EvaluationResult(
        episodes=5, finish_rate=len(laps) / 5,
        median_progress=1.0 if len(laps) == 5 else 0.9,
        mean_progress=1.0 if len(laps) == 5 else 0.9,
        best_lap_s=min(laps), median_lap_s=float(np.median(laps)),
        crash_rate=0.0, off_track_rate=0.0, stall_rate=0.0,
    )
    saved: list[str] = []
    monkeypatch.setattr("polybot.training.runner.evaluate_model", lambda *a, **k: result)
    monkeypatch.setattr(runner, "_emit", lambda event: None)
    monkeypatch.setattr(runner, "_save", lambda name, evaluation: saved.append(name) or tmp_path)
    runner._evaluate()
    assert expected in saved[0]


def test_promotion_uses_measured_frame_skip_reference(tmp_path, monkeypatch) -> None:
    config = replace(_config("grtqc"), output_root=tmp_path / "models")
    assert config.grtqc is not None
    config.grtqc.reference_lap_s = 24.616
    runner = TrainingRunner(config)
    runner.model = SimpleNamespace(num_timesteps=5000)
    result = EvaluationResult(5, 1.0, 1.0, 1.0, 24.5, 24.5, 0.0, 0.0, 0.0)
    saved = []
    monkeypatch.setattr("polybot.training.runner.evaluate_model", lambda *a, **k: result)
    monkeypatch.setattr(runner, "_emit", lambda event: None)
    monkeypatch.setattr(runner, "_save", lambda name, evaluation: saved.append(name) or tmp_path)
    runner._evaluate()
    assert saved == ["champion"]


def test_grtqc_keeps_contact_reduction_as_candidate_within_pace_tolerance(tmp_path, monkeypatch) -> None:
    config = replace(_config("grtqc"), output_root=tmp_path / "models")
    assert config.grtqc is not None
    config.grtqc.reference_lap_s = 24.616
    config.grtqc.champion_lap_tolerance_s = 0.15
    runner = TrainingRunner(config)
    runner.model = SimpleNamespace(num_timesteps=5000)
    previous = EvaluationResult(
        5, 1.0, 1.0, 1.0, 24.616, 24.616, 0.0, 0.0, 0.0,
        barrier_contact_steps=20,
    )
    current = EvaluationResult(
        5, 1.0, 1.0, 1.0, 24.893, 24.893, 0.0, 0.0, 0.0,
        barrier_contact_steps=10,
    )
    champion_dir = runner.registry.slot(config.track_name, "grtqc", "champion")
    champion_dir.mkdir(parents=True)
    (champion_dir / "metadata.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        runner.registry, "read_metadata",
        lambda _path: SimpleNamespace(
            training_config={"rewards": config.to_dict()["rewards"]},
            reward_semantics="executed-controls-v1", evaluation=previous.to_dict(),
        ),
    )
    monkeypatch.setattr("polybot.training.runner.evaluate_model", lambda *a, **k: current)
    monkeypatch.setattr(runner, "_emit", lambda event: None)
    saved = []
    monkeypatch.setattr(runner, "_save", lambda name, evaluation: saved.append(name) or tmp_path)
    runner._grtqc_weak_evaluations = 2

    runner._evaluate()

    assert saved == ["contact-candidate"]
    assert runner._grtqc_weak_evaluations == 0


@pytest.mark.parametrize(
    ("lap", "contacts", "expected"),
    [(25.1, 5, "contact-candidate"), (25.5, 5, "rejected"), (25.1, 10, "rejected")],
)
def test_cleaner_candidate_retains_faster_pace_without_adding_contacts(
    tmp_path, monkeypatch, lap, contacts, expected,
) -> None:
    config = replace(_config("grtqc"), output_root=tmp_path / "models")
    config.grtqc.contact_candidate_lap_tolerance_s = 1.5
    runner = TrainingRunner(config)
    runner.model = SimpleNamespace(num_timesteps=5000)
    saved_candidate = EvaluationResult(
        5, 1.0, 1.0, 1.0, 25.359, 25.359, 0.0, 0.0, 0.0, barrier_contact_steps=5,
    )
    current = replace(saved_candidate, best_lap_s=lap, median_lap_s=lap, barrier_contact_steps=contacts)
    candidate_dir = runner.registry.slot(config.track_name, "grtqc", "contact-candidate")
    candidate_dir.mkdir(parents=True)
    (candidate_dir / "metadata.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner.registry, "read_metadata", lambda path: SimpleNamespace(
        evaluation=saved_candidate.to_dict(),
    ))
    monkeypatch.setattr("polybot.training.runner.evaluate_model", lambda *a, **k: current)
    monkeypatch.setattr(runner, "_emit", lambda event: None)
    saved = []
    monkeypatch.setattr(runner, "_save", lambda name, evaluation: saved.append(name) or tmp_path)
    runner._evaluate()
    assert expected in saved[0]


def test_failed_candidate_restores_only_verified_actor_and_rewarms_critics(tmp_path, monkeypatch) -> None:
    config = replace(_config("grtqc"), output_root=tmp_path / "models")
    runner = TrainingRunner(config)
    current = th.nn.Linear(2, 2)
    verified = th.nn.Linear(2, 2)
    current.optimizer = th.optim.Adam(current.parameters(), lr=1e-5)
    verified.optimizer = th.optim.Adam(verified.parameters(), lr=1e-5)
    current(th.ones(1, 2)).sum().backward()
    current.optimizer.step()
    assert current.optimizer.state
    critic = th.nn.Linear(2, 1)
    preserved = {key: value.clone() for key, value in critic.state_dict().items()}
    runner.model = SimpleNamespace(
        actor=current, critic=critic, actor_lr=1e-5, actor_unlocked=True,
        critic_warmup_updates=2100,
        critic_updates_since_transfer=2100,
        _critic_loss_history=deque([1.0]), _disagreement_history=deque([0.1]),
        log_ent_coef=None, num_timesteps=9000,
    )
    runner.device = SimpleNamespace(resolved="cpu")
    monkeypatch.setattr(runner.backend, "load_model", lambda *a, **k: SimpleNamespace(actor=verified))
    events = []
    monkeypatch.setattr(runner, "_emit", events.append)
    result = EvaluationResult(5, 0.0, 0.5, 0.5, None, None, 1.0, 0.0, 0.0)
    runner._recover_grtqc_actor(result, tmp_path / "rejected")
    for key, value in current.state_dict().items():
        th.testing.assert_close(value, verified.state_dict()[key])
    for key, value in critic.state_dict().items():
        th.testing.assert_close(value, preserved[key])
    assert not current.optimizer.state
    assert runner.model.actor_lr == pytest.approx(1e-5)
    assert not runner.model.actor_unlocked
    assert runner.model.critic_updates_since_transfer == 1100
    assert events[0]["critic_replay_preserved"] is True
    assert events[0]["critic_cooldown_updates"] == 1000


def test_first_cleaner_candidate_is_saved_and_can_be_recovered_without_champion(tmp_path, monkeypatch) -> None:
    config = replace(_config("grtqc"), output_root=tmp_path / "models")
    runner = TrainingRunner(config)
    references = []
    runner.model = SimpleNamespace(num_timesteps=5000, set_actor_reference_observations=references.append)
    previous = EvaluationResult(5, 1.0, 1.0, 1.0, 24.263, 24.263, 0.0, 0.0, 0.0, barrier_contact_steps=10)
    current = replace(previous, best_lap_s=24.4, median_lap_s=24.4, barrier_contact_steps=5)
    initialization = runner.registry.slot(config.track_name, "grtqc", "initialization")
    candidate = runner.registry.slot(config.track_name, "grtqc", "contact-candidate")
    initialization.mkdir(parents=True)
    (initialization / "metadata.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(runner.registry, "read_metadata", lambda path: SimpleNamespace(
        evaluation=(previous if path == initialization else current).to_dict(),
    ))

    def evaluate(*args, observation_sink, **kwargs):
        observation_sink.append(np.zeros(3))
        return current

    def save(name, evaluation):
        assert name == "contact-candidate"
        candidate.mkdir(parents=True)
        (candidate / "metadata.json").write_text("{}", encoding="utf-8")
        return candidate

    monkeypatch.setattr("polybot.training.runner.evaluate_model", evaluate)
    monkeypatch.setattr(runner, "_emit", lambda event: None)
    monkeypatch.setattr(runner, "_save", save)
    runner._evaluate()
    assert runner._grtqc_verified_actor_source() == candidate
    assert len(references) == 1


@pytest.mark.parametrize("error", ["stale_episode: expired", "wrong_track: mismatch"])
def test_evaluation_retries_only_stale_episode_and_discards_partial_samples(tmp_path, monkeypatch, error) -> None:
    config = replace(_config("grtqc"), output_root=tmp_path / "models")
    runner = TrainingRunner(config)
    runner.model = SimpleNamespace(num_timesteps=5000)
    attempts = []

    def evaluate(*args, observation_sink, **kwargs):
        attempts.append(len(observation_sink))
        if len(attempts) == 1:
            observation_sink.append(np.zeros(3))
            raise ProtocolViolation(error)
        return EvaluationResult(5, 1.0, 1.0, 1.0, 24.263, 24.263, 0.0, 0.0, 0.0)

    monkeypatch.setattr("polybot.training.runner.evaluate_model", evaluate)
    monkeypatch.setattr(runner, "_emit", lambda event: None)
    monkeypatch.setattr(runner, "_save", lambda *args: tmp_path)
    if error.startswith("stale_episode:"):
        assert runner._evaluate().finish_rate == 1.0
        assert attempts == [0, 0]
    else:
        with pytest.raises(ProtocolViolation, match="wrong_track"):
            runner._evaluate()
        assert attempts == [0]


def test_grtqc_waits_for_three_weak_evaluations_before_recovery(tmp_path, monkeypatch) -> None:
    config = replace(_config("grtqc"), output_root=tmp_path / "models")
    runner = TrainingRunner(config)
    runner.model = SimpleNamespace(num_timesteps=5000, actor_unlocked=True)
    weak = EvaluationResult(5, 0.0, 0.5, 0.5, None, None, 1.0, 0.0, 0.0)
    monkeypatch.setattr("polybot.training.runner.evaluate_model", lambda *a, **k: weak)
    monkeypatch.setattr(runner, "_emit", lambda event: None)
    monkeypatch.setattr(runner, "_save", lambda name, evaluation: tmp_path / name)
    recovered = []
    monkeypatch.setattr(runner, "_recover_grtqc_actor", lambda *args: recovered.append(args))
    runner._evaluate()
    runner._evaluate()
    assert not recovered
    runner._evaluate()
    assert len(recovered) == 1


@pytest.mark.parametrize("actor_unlocked,screen_finished", [(True, False), (True, True), (False, True)])
def test_actor_screen_does_not_replace_five_lap_promotion(tmp_path, monkeypatch, actor_unlocked, screen_finished):
    config = replace(_config("grtqc"), output_root=tmp_path / "models")
    config.grtqc.screen_actor_evaluations = True
    config.grtqc.recovery_weak_evaluations = 5
    runner = TrainingRunner(config)
    runner.model = SimpleNamespace(num_timesteps=5000, actor_unlocked=actor_unlocked)
    calls, saved = [], []
    weak = EvaluationResult(1, 0.0, 0.5, 0.5, None, None, 1.0, 0.0, 0.0)
    finished = EvaluationResult(1, 1.0, 1.0, 1.0, 23.0, 23.0, 0.0, 0.0, 0.0)
    full = replace(finished, episodes=5)

    def evaluate(*args, episodes, **kwargs):
        calls.append(episodes)
        return (finished if screen_finished else weak) if episodes == 1 else full

    monkeypatch.setattr("polybot.training.runner.evaluate_model", evaluate)
    monkeypatch.setattr(runner, "_emit", lambda event: None)
    monkeypatch.setattr(runner, "_save", lambda name, evaluation: saved.append(name) or tmp_path / name)
    result = runner._evaluate()
    if actor_unlocked and not screen_finished:
        assert calls == [1]
        assert result.episodes == 1
        assert saved == ["checkpoints/step-5000-rejected"]
        assert runner._grtqc_weak_evaluations == 1
    else:
        assert calls == ([1, 5] if actor_unlocked else [5])
        assert result.episodes == 5
        assert saved == ["champion"]


def test_successful_screen_cannot_promote_failed_full_evaluation(tmp_path, monkeypatch):
    config = replace(_config("grtqc"), output_root=tmp_path / "models")
    config.grtqc.screen_actor_evaluations = True
    runner = TrainingRunner(config)
    runner.model = SimpleNamespace(num_timesteps=5000, actor_unlocked=True)
    finished = EvaluationResult(1, 1.0, 1.0, 1.0, 21.5, 21.5, 0.0, 0.0, 0.0)
    failed = EvaluationResult(5, 0.2, 0.5, 0.6, 21.5, 21.5, 0.8, 0.0, 0.0)
    monkeypatch.setattr(
        "polybot.training.runner.evaluate_model",
        lambda *args, episodes, **kwargs: finished if episodes == 1 else failed,
    )
    monkeypatch.setattr(runner, "_emit", lambda event: None)
    saved = []
    monkeypatch.setattr(runner, "_save", lambda name, evaluation: saved.append(name) or tmp_path / name)
    result = runner._evaluate()
    assert result == failed
    assert saved == ["checkpoints/step-5000-rejected"]
    assert not result.confirms_target_lap(22)


def test_denser_checks_keep_configured_recovery_opportunity(tmp_path, monkeypatch):
    config = replace(_config("grtqc"), output_root=tmp_path / "models")
    config.grtqc.recovery_weak_evaluations = 5
    runner = TrainingRunner(config)
    runner.model = SimpleNamespace(num_timesteps=5000, actor_unlocked=True)
    weak = EvaluationResult(5, 0.0, 0.5, 0.5, None, None, 1.0, 0.0, 0.0)
    monkeypatch.setattr("polybot.training.runner.evaluate_model", lambda *a, **k: weak)
    monkeypatch.setattr(runner, "_emit", lambda event: None)
    monkeypatch.setattr(runner, "_save", lambda name, evaluation: tmp_path / name)
    recovered = []
    monkeypatch.setattr(runner, "_recover_grtqc_actor", lambda *args: recovered.append(args))
    for _ in range(4):
        runner._evaluate()
    assert not recovered
    runner._evaluate()
    assert len(recovered) == 1


def test_rejected_grtqc_checkpoint_does_not_duplicate_replay(tmp_path, monkeypatch) -> None:
    config = replace(_config("grtqc"), output_root=tmp_path / "models")
    runner = TrainingRunner(config)
    runner.model = SimpleNamespace()
    monkeypatch.setattr(runner, "_metadata", lambda evaluation: None)
    monkeypatch.setattr(runner.registry, "write_metadata", lambda directory, metadata: None)
    flags = []
    monkeypatch.setattr(
        runner.backend, "save_model",
        lambda model, directory, *, resume: flags.append(resume),
    )
    runner._save("checkpoints/step-100-rejected")
    runner._save("latest")
    assert flags == [False, True]


def test_transport_disconnect_checkpoints_current_model(tmp_path, monkeypatch) -> None:
    config = replace(_config("grtqc"), output_root=tmp_path / "models")
    runner = TrainingRunner(config)
    runner.model = SimpleNamespace(num_timesteps=1234)
    saved = []
    events = []
    monkeypatch.setattr(runner, "_save", lambda name, evaluation=None: saved.append(name) or tmp_path)
    monkeypatch.setattr(runner, "_emit", events.append)

    runner._checkpoint_after_transport_failure(ConnectionError("player left race"))

    assert saved == ["latest"]
    assert events == [{
        "type": "transport_disconnected", "timesteps": 1234,
        "checkpoint": str(tmp_path), "error": "player left race",
    }]


def test_external_grtqc_initialization_is_kept_for_future_resumes(tmp_path) -> None:
    config = replace(_config("grtqc"), output_root=tmp_path / "models")
    runner = TrainingRunner(config)
    source = tmp_path / "prior-experiment" / "summer-1" / "grtqc" / "initialization"
    source.mkdir(parents=True)
    (source / "policy.zip").write_bytes(b"transferred-policy")
    (source / "metadata.json").write_text("{}", encoding="utf-8")
    (source / "transfer.json").write_text("{}", encoding="utf-8")

    preserved = runner._preserve_grtqc_initialization(source)

    assert preserved == runner.registry.slot(config.track_name, "grtqc", "initialization")
    assert (preserved / "policy.zip").read_bytes() == b"transferred-policy"
    assert (preserved / "transfer.json").is_file()
    assert runner._preserve_grtqc_initialization(source) == preserved

    (source / "policy.zip").write_bytes(b"different-policy")
    with pytest.raises(FileExistsError, match="GRTQC initialization differs"):
        runner._preserve_grtqc_initialization(source)


def test_raw_replay_retains_distinct_touchdown_demands_and_deterministic_runtime():
    config = _config("grtqc")
    config.grtqc.critic_raw_actions = True
    config.grtqc.critic_collection_std = 0
    model, env = _model(config)
    try:
        obs, _ = env.reset()
        obs[12], obs[17:21] = .5, 0
        model.policy_overlays = [{"kind": "air_brake", "start": .4, "end": .6, "duty": .25}]
        left = model._transform_action(np.array([.1, .4], dtype=np.float32), obs)
        left_base = model._air_brake_base_action.copy()
        right = model._transform_action(np.array([.1, .8], dtype=np.float32), obs)
        right_base = model._air_brake_base_action.copy()
        np.testing.assert_array_equal(left, right)
        assert left_base[1] != right_base[1]  # post-overlay labels alias touchdown behavior
        model._last_obs = obs[None]
        expected, _ = model.predict(obs[None], deterministic=True)
        executed, recorded = model._sample_action(0)
        np.testing.assert_array_equal(executed, expected)
        raw, _ = model.policy.predict(obs[None], deterministic=True)
        np.testing.assert_array_equal(recorded, raw)
        assert recorded[0, 1] != executed[0, 1]
        with th.no_grad():
            demand = th.tensor([[.1, .8]])
            th.testing.assert_close(model._critic_actions(demand, th.as_tensor(obs[None])), demand)
        changed = replace(config.grtqc, critic_raw_actions=False)
        with pytest.raises(ValueError, match="action semantics"):
            GRTQCBackend().configure_resume(model, replace(config, grtqc=changed), "cpu")
    finally:
        env.close()


def test_pace_acceptance_rejects_a_slower_clean_policy_immediately(tmp_path, monkeypatch):
    config = replace(_config("grtqc"), output_root=tmp_path / "models")
    config.grtqc.pace_only_actor_acceptance = True
    config.grtqc.recovery_weak_evaluations = 1
    runner = TrainingRunner(config)
    runner.model = SimpleNamespace(num_timesteps=5000, actor_unlocked=True)
    slower = EvaluationResult(5, 1., 1., 1., 24.485, 24.485, 0., 0., 0., barrier_contact_steps=5)
    monkeypatch.setattr("polybot.training.runner.evaluate_model", lambda *a, **k: slower)
    monkeypatch.setattr(runner, "_emit", lambda event: None)
    saved, recovered = [], []
    monkeypatch.setattr(runner, "_save", lambda name, evaluation: saved.append(name) or tmp_path / name)
    monkeypatch.setattr(runner, "_recover_grtqc_actor", lambda *args: recovered.append(args))
    runner._evaluate()
    assert saved == ["checkpoints/step-5000-rejected"]
    assert len(recovered) == 1
    assert runner._grtqc_verified_actor_source().name == "initialization"


def test_actor_update_diagnostics_survive_logger_dump_and_model_roundtrip(tmp_path):
    model, env = _model(_config("grtqc"))
    try:
        model.set_logger(configure(None, []))
        model.learn(16)
        model.actor_unlocked = True
        model.train(1, 8)
        metrics = GRTQCBackend().metrics(model)
        assert metrics["actor_gradient_norm"] > 0
        assert metrics["critic_gradient_norm"] > 0
        assert metrics["actor_layer_update_norms"]
        assert metrics["actor_adam_steps"] == 1
        model.logger.dump()
        assert GRTQCBackend().metrics(model)["actor_gradient_norm"] == metrics["actor_gradient_norm"]
        model.save(tmp_path / "policy")
        restored = GRTQCBackend().load_model(tmp_path / "policy.zip", None, "cpu")
        restored.set_logger(configure(None, []))
        assert GRTQCBackend().metrics(restored)["actor_gradient_norm"] == metrics["actor_gradient_norm"]
        assert all(float(state["step"]) == 1 for state in restored.actor.optimizer.state.values())
    finally:
        env.close()


def test_evaluation_replay_keeps_real_terminal_observations_and_marks_reset_boundary():
    config = _config("grtqc")
    runner = TrainingRunner(config)
    model, env = _model(config)
    runner.model = model
    runner._emit = lambda event: None
    try:
        first = np.zeros(105, dtype=np.float32)
        terminal = np.ones(105, dtype=np.float32)
        model.replay_buffer.add(first[None], first[None], np.zeros((1, 2)), np.zeros(1), [False], [{}])
        runner._retain_grtqc_evaluation_transitions([{
            "observation": first, "next_observation": terminal,
            "action": np.array([.2, .6], dtype=np.float32), "reward": 123.,
            "done": True, "timeout": False,
        }])
        assert model.replay_buffer.dones[0, 0] == 1
        assert model.replay_buffer.timeouts[0, 0] == 1
        assert model.replay_buffer.dones[1, 0] == 1
        assert model.replay_buffer.timeouts[1, 0] == 0
        np.testing.assert_array_equal(model.replay_buffer.next_observations[1, 0], terminal)
        assert model.replay_buffer.rewards[1, 0] == pytest.approx(123 * config.reward_scale)
        model.actor_verified_state_sampling = True
        model.actor_unlocked = True
        model.set_actor_reference_observations(np.stack([first, terminal]))
        model.set_logger(configure(None, []))
        model.train(1, 2)
        assert model._training_diagnostics["train/actor_update_norm"] > 0
        assert model._training_diagnostics["train/critic_update_norm"] > 0
    finally:
        env.close()


def test_pace_profile_ranks_the_measured_faster_complete_lap_above_the_cleaner_lap():
    records = json.loads(Path("tests/fixtures/grtqc-complete-lap-rewards.json").read_text())
    profile = json.loads(Path("profiles/training/summer-1-grtqc-causal-30.json").read_text())
    totals = []
    for record in records:
        terms = dict(record["reward_terms"])
        for key in ("on_track_speed", "speed_pace", "unsafe_speed", "ground_brake", "airborne_brake"):
            terms[key] = 0.
        terms["elapsed"] *= profile["rewards"]["elapsed_cost_per_s"] / record["recorded_profile"]["elapsed_cost_per_s"]
        terms["barrier_contact"] *= (
            profile["rewards"]["barrier_contact_penalty"] / record["recorded_profile"]["barrier_contact_penalty"]
        )
        totals.append(sum(terms.values()))
    assert records[0]["lap_s"] == 24.263
    assert records[1]["lap_s"] == 24.485
    assert sum(records[0]["reward_terms"].values()) < sum(records[1]["reward_terms"].values())
    assert totals[0] > totals[1]
    assert profile["rewards"]["barrier_contact_penalty"] < 0


def test_resume_does_not_deploy_an_unverified_actor_from_a_transport_checkpoint(tmp_path, monkeypatch):
    config = _config("grtqc")
    config.grtqc.pace_only_actor_acceptance = True
    model, env = _model(config)
    verified, reference_env = _model(config)
    runner = TrainingRunner(config)
    runner.model = model
    try:
        model.set_logger(configure(None, []))
        model.learn(16)
        model.actor_unlocked = True
        model.train(1, 8)
        assert model.actor.optimizer.state
        critic = {name: value.clone() for name, value in model.critic.state_dict().items()}
        replay = model.replay_buffer
        saved, events = [], []
        monkeypatch.setattr(runner, "_save", lambda name: saved.append(name) or tmp_path / name)
        monkeypatch.setattr(runner, "_emit", events.append)
        runner._restore_unverified_grtqc_resume(verified, tmp_path / "initialization")
        assert saved and saved[0].endswith("resume-rejected")
        for name, value in model.actor.state_dict().items():
            th.testing.assert_close(value, verified.actor.state_dict()[name], rtol=0, atol=0)
        for name, value in model.critic.state_dict().items():
            th.testing.assert_close(value, critic[name], rtol=0, atol=0)
        assert not model.actor.optimizer.state
        assert not model.ent_coef_optimizer.state
        th.testing.assert_close(model.log_ent_coef, verified.log_ent_coef)
        assert model.replay_buffer is replay and replay.size() == 16
        assert model.critic.optimizer.state
        assert events[0]["critic_replay_preserved"]
        runner._restore_unverified_grtqc_resume(verified, tmp_path / "initialization")
        assert len(saved) == 1
    finally:
        env.close()
        reference_env.close()


def test_scratch_uses_fresh_actor_full_state_and_real_stochastic_exploration(tmp_path):
    config = _config("grtqc")
    config.grtqc.training_origin = "scratch"
    config.grtqc.critic_controller_state = config.grtqc.critic_environment_state = True
    config.grtqc.critic_raw_actions = True
    model, env = _model(config)
    teacher, teacher_env = _model(_config("tqc"))
    try:
        obs, _ = env.reset()
        assert model.replay_buffer.size() == 0
        assert not model.policy_overlays and not model.speed_bias_schedule
        assert model.actor.features_extractor.features_dim == 121
        assert model.policy.actor_gate_scale == 1.
        assert not th.equal(model.actor.latent_pi[0].weight[:, :105], teacher.actor.latent_pi[0].weight)
        model.num_timesteps = model.learning_starts + 1
        model._last_obs = obs[None]
        samples = np.stack([model._sample_action(0)[0][0] for _ in range(12)])
        assert samples.std(axis=0).min() > .05
        model.save(tmp_path / "scratch")
        restored = GRTQCBackend().load_model(tmp_path / "scratch.zip", None, "cpu")
        assert restored.training_origin == "scratch"
        assert restored.actor.features_extractor.features_dim == 121
        assert restored.policy.actor_gate_scale == 1.
        with pytest.raises(ValueError, match="scratch experiment"):
            GRTQCBackend().configure_resume(model, _config("grtqc"), "cpu", fresh_replay=True)
    finally:
        env.close()
        teacher_env.close()


def test_delayed_actor_updates_count_across_single_update_training_calls():
    config = _config("grtqc")
    config.grtqc.actor_update_interval = 2
    model, env = _model(config)
    try:
        model.set_logger(configure(None, []))
        model.learn(16)
        model.actor_unlocked = True
        model._n_updates = 0
        before = {name: value.clone() for name, value in model.actor.state_dict().items()}
        model.train(1, 8)
        assert all(th.equal(value, before[name]) for name, value in model.actor.state_dict().items())
        model.train(1, 8)
        assert any(not th.equal(value, before[name]) for name, value in model.actor.state_dict().items())
        assert model._training_diagnostics["train/actor_adam_steps"] == 1
    finally:
        env.close()


def test_scratch_promotes_first_reliable_lap_without_the_tqc_time_ceiling(tmp_path, monkeypatch):
    config = replace(_config("grtqc"), output_root=tmp_path / "scratch")
    config.grtqc.training_origin = "scratch"
    runner = TrainingRunner(config)
    runner.model = SimpleNamespace(num_timesteps=12000, actor_unlocked=True)
    reliable = EvaluationResult(5, 1., 1., 1., 27., 27., 0., 0., 0.)
    monkeypatch.setattr("polybot.training.runner.evaluate_model", lambda *a, **k: reliable)
    events, saved = [], []
    monkeypatch.setattr(runner, "_emit", events.append)
    monkeypatch.setattr(runner, "_save", lambda name, evaluation: saved.append(name) or tmp_path / name)
    runner._evaluate()
    assert saved == ["champion"]
    assert runner.model.scratch_stage == "pace"
    assert config.grtqc.target_entropy == config.grtqc.scratch_pace_target_entropy
    assert not any(event["type"] == "pace_milestone" for event in events)


def test_scratch_rejected_candidate_keeps_learning_without_actor_recovery(tmp_path, monkeypatch):
    config = replace(_config("grtqc"), output_root=tmp_path / "scratch")
    config.grtqc.training_origin = "scratch"
    config.grtqc.recovery_weak_evaluations = 1
    runner = TrainingRunner(config)
    runner.model = SimpleNamespace(num_timesteps=12000, actor_unlocked=True)
    failed = EvaluationResult(1, 0., .65, .65, None, None, 1., 0., 0.)
    monkeypatch.setattr("polybot.training.runner.evaluate_model", lambda *a, **k: failed)
    monkeypatch.setattr(runner, "_emit", lambda event: None)
    monkeypatch.setattr(runner, "_save", lambda name, evaluation: tmp_path / name)
    monkeypatch.setattr(runner, "_recover_grtqc_actor", lambda *a: pytest.fail("scratch candidate was reset"))
    for _ in range(6):
        runner._evaluate()
    assert runner.model.actor_unlocked
    assert runner._grtqc_weak_evaluations == 0


def test_scratch_training_saves_a_random_origin_and_resumes_without_teacher(tmp_path):
    config = replace(_config("grtqc"), output_root=tmp_path / "scratch", log_root=tmp_path / "logs", timesteps=16)
    config.grtqc.training_origin = "scratch"
    config.grtqc.critic_warmup_updates = 1
    config.grtqc.critic_readiness_window = 2
    runner = TrainingRunner(config)
    latest = runner.run()
    origin = runner.registry.slot(config.track_name, "grtqc", "initialization")
    assert (origin / "policy.zip").is_file()
    assert not (origin / "transfer.json").exists()
    metadata = runner.registry.read_metadata(origin)
    assert metadata.training_config["grtqc"]["training_origin"] == "scratch"
    assert metadata.training_timesteps == 0
    second = TrainingRunner(replace(config, timesteps=8))
    second.run(resume=latest)
    assert second.model.num_timesteps == 24
    assert second.model.training_origin == "scratch"
    with pytest.raises(FileExistsError, match="scratch output already exists"):
        TrainingRunner(config).run()


def test_parallel_ports_are_explicit_and_do_not_change_transfer_default():
    scratch = TrainingConfig.from_dict(json.loads(Path("profiles/training/summer-1-grtqc-scratch-30.json").read_text()))
    assert TrainingRunner(scratch)._transport().endpoint == "ws://127.0.0.1:8766"
    assert TrainingRunner(replace(_config("grtqc"), backend="websocket"))._transport().endpoint == "ws://127.0.0.1:8765"
    assert scratch.grtqc.target_lap_s == 22.
    assert scratch.grtqc.training_origin == "scratch"
    assert scratch.curriculum.phases[0].mode == "quarters-randomised"
    assert scratch.rewards.guidance_reward_scale == 0
    # At frame skip 30, observations arrive every ~0.5s. Allow correction
    # after a sampled airborne roll and charge the same terminal cost as crashes.
    assert scratch.rewards.airborne_roll_timeout_s >= 1.0
    assert scratch.rewards.airborne_roll_failure_penalty == scratch.rewards.crash_penalty
    assert scratch.rewards.barrier_contact_penalty <= -200.0
    assert scratch.grtqc.actor_learning_rate == 1e-4
    assert scratch.rewards.speed_pace_reward_per_m_per_mps == 0.01
    assert scratch.rewards.early_off_track_penalty == scratch.rewards.off_track_penalty == -400.
    assert scratch.rewards.finish_fast_bonus > 2 * scratch.rewards.finish_bonus
    with pytest.raises(ValueError, match="WebSocket port"):
        replace(scratch, websocket_port=0)
