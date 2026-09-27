"""Expand a six-action DQN checkpoint to nine actions while preserving experience."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import DQN

from polybot.algorithms.dqn import DQNBackend
from polybot.models.registry import ModelRegistry
from polybot.training.config import TrainingConfig

SIX_TO_NINE = (0, 1, 3, 4, 6, 7)
BRAKE_ROWS = ((2, 0), (5, 3), (8, 6))


def _expand_q_net(source: Any, target: Any) -> None:
    old_layers = source.q_net
    new_layers = target.q_net
    if len(old_layers) != len(new_layers):
        raise ValueError("DQN network architecture changed between brake stages")
    with torch.no_grad():
        for old, new in zip(old_layers[:-1], new_layers[:-1], strict=True):
            new.load_state_dict(old.state_dict())
        old_head, new_head = old_layers[-1], new_layers[-1]
        if old_head.out_features != 6 or new_head.out_features != 9:
            raise ValueError("brake stage requires a six-output source and nine-output target")
        for old_row, new_row in enumerate(SIX_TO_NINE):
            new_head.weight[new_row].copy_(old_head.weight[old_row])
            new_head.bias[new_row].copy_(old_head.bias[old_row])
        for brake_row, coast_row in BRAKE_ROWS:
            new_head.weight[brake_row].copy_(new_head.weight[coast_row])
            new_head.bias[brake_row].copy_(new_head.bias[coast_row] - 0.25)


def expand_no_brake_checkpoint(
    source_slot: Path, target_config: TrainingConfig, env: Any, device: str
) -> Path:
    """Create a resumable nine-action latest slot from a six-action latest slot."""
    if target_config.algorithm != "dqn" or target_config.dqn is None:
        raise ValueError("brake transfer requires DQN configuration")
    if target_config.dqn.action_set != "full":
        raise ValueError("target DQN must enable brake")
    backend = DQNBackend()
    source_meta = ModelRegistry().read_metadata(source_slot)
    source_config = TrainingConfig.from_dict(source_meta.training_config)
    if source_config.dqn is None or source_config.dqn.action_set != "no_brake":
        raise ValueError("source DQN must use no_brake actions")
    if source_meta.action_schema != "digital-discrete-6-no-brake-v2":
        raise ValueError("source action schema is not the six-action DQN schema")
    for key in ("track_id", "lookahead_count", "frame_skip", "reward_scale"):
        if getattr(source_config, key) != getattr(target_config, key):
            raise ValueError(f"brake transfer requires matching {key}")
    if source_config.rewards != target_config.rewards:
        raise ValueError("brake transfer requires unchanged replay rewards")
    if source_config.dqn.architecture != target_config.dqn.architecture:
        raise ValueError("brake transfer requires matching Q-network architecture")

    source = DQN.load(str(source_slot / "policy.zip"), device=device)
    source.load_replay_buffer(str(source_slot / "replay.pkl"))
    target = backend.create_model(target_config, env, device)
    _expand_q_net(source.policy.q_net, target.policy.q_net)
    _expand_q_net(source.policy.q_net_target, target.policy.q_net_target)
    replay = source.replay_buffer
    if replay is None:
        raise ValueError("source DQN checkpoint has no replay")
    original = replay.actions.astype(np.int64, copy=True)
    if np.any((original < 0) | (original >= 6)):
        raise ValueError("source replay contains an action outside the six-action set")
    replay.actions[...] = np.take(np.asarray(SIX_TO_NINE), original)
    replay.action_space = spaces.Discrete(9)
    target.replay_buffer = replay
    target.num_timesteps = source.num_timesteps
    target._n_updates = source._n_updates
    target._n_calls = source._n_calls

    registry = ModelRegistry(target_config.output_root)
    destination = registry.slot(target_config.track_name, "dqn", "latest")
    backend.save_model(target, destination, resume=True)
    counts = backend.parameter_counts(target)
    registry.write_metadata(destination, replace(
        source_meta,
        action_schema="digital-discrete-9-v2",
        actor_parameters=counts["actor"],
        critic_parameters=counts["critic"],
        total_trainable_parameters=counts["total"],
        training_config=target_config.to_dict(),
        evaluation=None,
    ))
    return destination
