from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pytest
import torch
from sb3_contrib import QRDQN
from stable_baselines3 import DQN

from polybot.algorithms.registry import backend_for
from polybot.control.native_digital import NativeDigitalActionAdapter
from polybot.environment.env import PolyTrackEnv
from polybot.environment.observations import SCHEMA as OBSERVATION_SCHEMA
from polybot.mock import MockSimulatorTransport
from polybot.models.registry import ModelMetadata, ModelRegistry
from polybot.training.config import DQNConfig, EvaluationConfig, TrainingConfig
from polybot.training.dqn_qr_migration import migrate_dqn_to_qr
from polybot.training.runner import ScaledTrainingReward, TrainingRunner


def test_legacy_dqn_weights_and_replay_continue_as_qr_dqn(tmp_path) -> None:
    legacy_config = TrainingConfig(
        algorithm="dqn", device="cpu", timesteps=32,
        dqn=DQNConfig(architecture="yosh_2020", learning_starts=8, batch_size=8,
                      replay_capacity=128),
        evaluation=EvaluationConfig(32, 1), checkpoint_interval=0,
        output_root=tmp_path / "legacy", log_root=tmp_path / "logs",
    )
    legacy_slot = ModelRegistry(legacy_config.output_root).slot(
        legacy_config.track_name, "dqn", "latest"
    )
    env = ScaledTrainingReward(PolyTrackEnv(
        MockSimulatorTransport(), track_id=legacy_config.track_id,
        action_adapter=NativeDigitalActionAdapter(),
    ), legacy_config.reward_scale)
    try:
        old = DQN(
            "MlpPolicy", env, device="cpu", seed=0, verbose=0,
            learning_starts=8, batch_size=8, buffer_size=128,
            policy_kwargs={"net_arch": [64, 16]},
        )
        old.learn(32)
        legacy_slot.mkdir(parents=True)
        old.save(legacy_slot / "policy.zip")
        old.save_replay_buffer(legacy_slot / "replay.pkl")
    finally:
        env.close()
    ModelRegistry().write_metadata(legacy_slot, ModelMetadata(
        algorithm="dqn", architecture="yosh_2020", actor_parameters=0,
        critic_parameters=sum(p.numel() for p in old.policy.q_net.parameters()),
        total_trainable_parameters=sum(p.numel() for p in old.policy.q_net.parameters()),
        observation_schema=OBSERVATION_SCHEMA, action_schema="digital-discrete-9-v2",
        track_name=legacy_config.track_name, track_id=legacy_config.track_id,
        lookahead_count=legacy_config.lookahead_count, reward_profile=None,
        curriculum=asdict(legacy_config.curriculum),
        training_config=legacy_config.to_dict(), training_timesteps=old.num_timesteps,
        simulator_ticks=old.num_timesteps * legacy_config.frame_skip,
        wall_seconds=0.0, seed=0, device="cpu", finishes=0, crashes=0,
    ))

    target_config = replace(
        legacy_config, output_root=tmp_path / "qr", timesteps=16,
        evaluation=EvaluationConfig(16, 1),
    )
    target_env = PolyTrackEnv(
        MockSimulatorTransport(), track_id=target_config.track_id,
        action_adapter=NativeDigitalActionAdapter(),
    )
    try:
        backend = backend_for("dqn")
        assert isinstance(backend.load_model(legacy_slot / "policy.zip", target_env, "cpu"), DQN)
        with pytest.raises(ValueError, match="migrate"):
            backend.load_model(legacy_slot / "policy.zip", target_env, "cpu", resume=True)
        migrated = migrate_dqn_to_qr(legacy_slot, target_config, target_env, "cpu")
        converted = QRDQN.load(migrated / "policy.zip", device="cpu")
        observations = torch.randn(8, old.observation_space.shape[0])
        with torch.no_grad():
            old_q = old.policy.q_net(observations)
            new_q = converted.policy.quantile_net(observations).mean(dim=1)
        np.testing.assert_allclose(new_q.numpy(), old_q.numpy(), atol=1e-6)
        converted.load_replay_buffer(migrated / "replay.pkl")
        assert converted.replay_buffer.size() > 0
        assert converted.num_timesteps == old.num_timesteps
        assert ModelRegistry().read_metadata(migrated).implementation == "qr_dqn"
        with pytest.raises(FileExistsError, match="overwrite"):
            migrate_dqn_to_qr(legacy_slot, target_config, target_env, "cpu")
    finally:
        target_env.close()
    resumed = TrainingRunner(target_config).run(resume=migrated)
    assert QRDQN.load(resumed / "policy.zip", device="cpu").num_timesteps > old.num_timesteps
