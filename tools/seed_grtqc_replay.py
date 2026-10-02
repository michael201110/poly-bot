"""Append compatible TQC experience to a separate GRTQC resume checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3.common.save_util import load_from_pkl

from polybot.algorithms.grtqc import GRTQCBackend
from polybot.models.registry import ModelRegistry
from polybot.training.config import TrainingConfig


def seed_replay(source: Path, current: Path, destination: Path) -> dict[str, object]:
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    source_meta = ModelRegistry(source.parents[2]).read_metadata(source)
    current_meta = ModelRegistry(current.parents[2]).read_metadata(current)
    if source_meta.algorithm != "tqc" or current_meta.algorithm != "grtqc":
        raise ValueError("expected a TQC source and GRTQC continuation")
    source_config = TrainingConfig.from_dict(source_meta.training_config)
    current_config = TrainingConfig.from_dict(current_meta.training_config)
    shared = ("track_id", "track_name", "frame_skip", "lookahead_count", "reward_scale")
    if any(getattr(source_config, key) != getattr(current_config, key) for key in shared):
        raise ValueError("source and continuation differ in track, timing, or reward scale")
    if asdict(source_config.rewards) != asdict(current_config.rewards):
        raise ValueError("source and continuation rewards differ after default normalization")
    if (
        source_meta.observation_schema != current_meta.observation_schema
        or source_meta.action_schema != current_meta.action_schema
        or source_meta.reward_semantics != current_meta.reward_semantics
        or source_meta.action_semantics != current_meta.action_semantics
    ):
        raise ValueError("source and continuation transition semantics differ")
    backend = GRTQCBackend()
    model = backend.load_model(current / "policy.zip", None, "cpu", resume=True)
    target = model.replay_buffer
    source_replay = load_from_pkl(source / "replay.pkl")
    if target is None or source_replay is None:
        raise RuntimeError("both replay buffers are required")
    if (
        target.n_envs != source_replay.n_envs
        or target.n_envs != 1
        or target.optimize_memory_usage
        or source_replay.optimize_memory_usage
        or target.handle_timeout_termination != source_replay.handle_timeout_termination
    ):
        raise ValueError("incompatible replay storage layout")
    names = ("observations", "next_observations", "actions", "rewards", "dones", "timeouts")
    if any(
        getattr(target, key).shape[1:] != getattr(source_replay, key).shape[1:]
        or getattr(target, key).dtype != getattr(source_replay, key).dtype
        for key in names
    ):
        raise ValueError("replay array shape or dtype differs")
    original_count = target.size()
    imported_count = source_replay.size()
    if target.full or source_replay.full or original_count + imported_count >= target.buffer_size:
        raise ValueError("replay merge requires two non-full buffers with spare capacity")
    if original_count and imported_count and not target.dones[original_count - 1, 0]:
        target.dones[original_count - 1, 0] = target.timeouts[original_count - 1, 0] = 1
    for key in names:
        getattr(target, key)[original_count:original_count + imported_count] = (
            getattr(source_replay, key)[:imported_count]
        )
    target.pos = original_count + imported_count
    model.invalidate_critic_reference()
    model.critic_mc_updates_done = 0
    model.actor_unlocked = False
    model.critic_updates_since_transfer = 0
    model._critic_loss_history.clear()
    model._disagreement_history.clear()
    destination.mkdir(parents=True)
    backend.save_model(model, destination, resume=True)
    saved = backend.load_model(destination / "policy.zip", None, "cpu", resume=True)
    if saved.replay_buffer is None or saved.replay_buffer.size() != target.size():
        raise RuntimeError("saved replay failed round-trip validation")
    if any(
        not th.equal(value, saved.actor.state_dict()[key])
        for key, value in model.actor.state_dict().items()
    ):
        raise RuntimeError("saved actor differs from continuation actor")
    ModelRegistry(current.parents[2]).write_metadata(destination, current_meta)
    report: dict[str, object] = {
        "source": str(source),
        "source_policy_sha256": hashlib.sha256((source / "policy.zip").read_bytes()).hexdigest(),
        "continuation": str(current),
        "continuation_policy_sha256": hashlib.sha256((current / "policy.zip").read_bytes()).hexdigest(),
        "destination": str(destination),
        "original_transitions": original_count,
        "imported_transitions": imported_count,
        "combined_transitions": target.size(),
        "imported_terminal_transitions": int(np.sum(source_replay.dones[:imported_count])),
        "imported_high_reward_transitions": int(np.sum(source_replay.rewards[:imported_count] > 10)),
        "actor_unchanged": True,
        "critic_warmup_reset": True,
    }
    (destination / "replay-source.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "models/v2-dqn-qr-migrated-20260927/summer-1/tqc/champion"
    ))
    parser.add_argument("--current", type=Path, default=Path("models/summer-1/grtqc/latest"))
    parser.add_argument("--destination", type=Path, default=Path(
        "models/summer-1/grtqc/replay-seeded-20261001"
    ))
    args = parser.parse_args()
    print(json.dumps(seed_replay(args.source, args.current, args.destination), indent=2))


if __name__ == "__main__":
    main()
