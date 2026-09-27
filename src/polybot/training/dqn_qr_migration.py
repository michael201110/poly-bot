"""Convert a saved vanilla DQN into a resumable QR-DQN checkpoint."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from stable_baselines3 import DQN

from polybot.algorithms.dqn import DQNBackend
from polybot.models.registry import ModelRegistry
from polybot.training.config import TrainingConfig


def _copy_network(source: Any, target: Any, n_actions: int, n_quantiles: int) -> None:
    old_layers = source.q_net
    new_layers = target.quantile_net
    if len(old_layers) != len(new_layers):
        raise ValueError("DQN and QR-DQN hidden architectures differ")
    with torch.no_grad():
        for old, new in zip(old_layers[:-1], new_layers[:-1], strict=True):
            new.load_state_dict(old.state_dict())
        old_head, new_head = old_layers[-1], new_layers[-1]
        if old_head.out_features != n_actions or new_head.out_features != n_actions * n_quantiles:
            raise ValueError("DQN and QR-DQN action heads are incompatible")
        for quantile in range(n_quantiles):
            rows = slice(quantile * n_actions, (quantile + 1) * n_actions)
            new_head.weight[rows].copy_(old_head.weight)
            new_head.bias[rows].copy_(old_head.bias)


def migrate_dqn_to_qr(
    source_slot: Path, target_config: TrainingConfig, env: Any, device: str
) -> Path:
    """Preserve hidden weights, greedy Q-values, replay, and step count in a new slot."""
    if target_config.algorithm != "dqn" or target_config.dqn is None:
        raise ValueError("QR-DQN migration requires a DQN training configuration")
    source_meta = ModelRegistry().read_metadata(source_slot)
    if source_meta.algorithm != "dqn" or source_meta.implementation is not None:
        raise ValueError("source must be a legacy vanilla DQN checkpoint")
    source_config = TrainingConfig.from_dict(source_meta.training_config)
    if source_config.dqn is None:
        raise ValueError("source DQN configuration is missing")
    for key in ("track_id", "lookahead_count", "frame_skip", "reward_scale"):
        if getattr(source_config, key) != getattr(target_config, key):
            raise ValueError(f"QR-DQN migration requires matching {key}")
    if source_config.rewards != target_config.rewards:
        raise ValueError("QR-DQN migration requires unchanged replay rewards")
    if source_config.dqn.architecture != target_config.dqn.architecture:
        raise ValueError("QR-DQN migration requires matching hidden architecture")
    if source_config.dqn.action_set != target_config.dqn.action_set:
        raise ValueError("QR-DQN migration requires matching action sets")

    registry = ModelRegistry(target_config.output_root)
    destination = registry.slot(target_config.track_name, "dqn", "latest")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing QR-DQN checkpoint: {destination}")
    source = DQN.load(str(source_slot / "policy.zip"), device=device)
    replay_path = source_slot / "replay.pkl"
    if not replay_path.is_file():
        raise FileNotFoundError(f"DQN migration requires replay state: {replay_path}")
    target_backend = DQNBackend()
    target = target_backend.create_model(target_config, env, device)
    n_actions = int(target.action_space.n)
    n_quantiles = target_config.dqn.n_quantiles
    _copy_network(source.policy.q_net, target.policy.quantile_net, n_actions, n_quantiles)
    _copy_network(source.policy.q_net_target, target.policy.quantile_net_target, n_actions, n_quantiles)
    target.load_replay_buffer(str(replay_path))
    target.num_timesteps = source.num_timesteps
    target._n_updates = source._n_updates
    target._n_calls = source._n_calls
    counts = target_backend.parameter_counts(target)
    target_backend.save_model(target, destination, resume=True)
    registry.write_metadata(destination, replace(
        source_meta,
        actor_parameters=counts["actor"],
        critic_parameters=counts["critic"],
        total_trainable_parameters=counts["total"],
        training_config=target_config.to_dict(),
        evaluation=None,
        implementation="qr_dqn",
    ))
    return destination
