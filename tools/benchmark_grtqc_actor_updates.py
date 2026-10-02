"""Compare saved actor updates on one real controller-state replay checkpoint.

Snapshots are diagnostic policies, not resumable training checkpoints. Live
completion and timing must be checked separately before using any candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3.common.logger import configure

from polybot.algorithms.grtqc import GRTQCBackend
from polybot.models.registry import ModelRegistry
from polybot.training.config import TrainingConfig


def benchmark(learner: Path, actor: Path, config: TrainingConfig, output: Path,
              critic_updates: int, snapshots: list[int]) -> dict:
    if output.exists():
        raise FileExistsError(f"preserve existing diagnostic: {output}")
    if critic_updates < 0 or not snapshots or min(snapshots) < 1:
        raise ValueError("diagnostic update counts must be nonnegative with positive actor snapshots")
    th.set_num_threads(1)
    backend = GRTQCBackend()
    registry = ModelRegistry(config.output_root)
    for directory in (learner, actor):
        registry.validate(registry.read_metadata(directory), config, backend.action_adapter(config).schema)
    model = backend.load_model(learner / "policy.zip", None, "cpu", resume=True)
    verified = backend.load_model(actor / "policy.zip", None, "cpu")
    backend.configure_resume(model, config, "cpu")
    backend.restore_actor_weights(model, verified)
    model.actor.optimizer.state.clear()
    model.set_logger(configure(None, []))
    replay = model.replay_buffer
    if replay is None or not replay.size():
        raise ValueError("diagnostic needs real saved replay")
    model.set_actor_reference_observations(replay.observations[:min(809, replay.size()), 0])
    initial = {name: value.clone() for name, value in model.actor.state_dict().items()}
    model.actor_unlocked = False
    model.critic_warmup_updates = 10_000_000
    np.random.seed(17)
    th.manual_seed(17)
    if critic_updates:
        model.train(critic_updates, config.grtqc.batch_size)
    if any(not th.equal(value, initial[name]) for name, value in model.actor.state_dict().items()):
        raise RuntimeError("critic-only diagnostic changed the actor")
    model.actor_unlocked = True
    output.mkdir(parents=True)
    previous = 0
    records = []
    for updates in sorted(set(snapshots)):
        model.train(updates - previous, config.grtqc.batch_size)
        previous = updates
        directory = output / f"updates-{updates}"
        backend.save_model(model, directory)
        record = {"actor_updates": updates, "policy": str(directory / "policy.zip"),
                  "metrics": backend.metrics(model)}
        records.append(record)
        print(json.dumps(record), flush=True)
    result = {
        "learner": str(learner), "actor": str(actor), "config": config.to_dict(),
        "critic_only_updates": critic_updates, "actor_unchanged_during_warmup": True,
        "snapshots": records, "live_validation": "pending",
    }
    (output / "diagnostic.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learner", type=Path, required=True)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--critic-updates", type=int, default=2000)
    parser.add_argument("--snapshots", type=int, nargs="+", default=[1, 32, 64, 96, 128, 192, 256])
    args = parser.parse_args()
    config = TrainingConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8-sig")))
    if config.algorithm != "grtqc":
        parser.error("requires a GRTQC training config")
    benchmark(args.learner, args.actor, config, args.output, args.critic_updates, args.snapshots)


if __name__ == "__main__":
    main()
