"""Compare measured mock training TPS across algorithms and network sizes."""

from __future__ import annotations

import argparse
import json
import time

import torch

from polybot.algorithms.registry import backend_for
from polybot.environment.env import PolyTrackEnv
from polybot.mock import MockSimulatorTransport
from polybot.training.config import DQNConfig, PPOConfig, TQCConfig, TrainingConfig
from polybot.training.devices import resolve_device
from polybot.training.runner import ScaledTrainingReward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--algorithms", nargs="+", choices=("ppo", "dqn", "tqc"),
                        default=("ppo", "dqn", "tqc"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    for algorithm in args.algorithms:
        backend = backend_for(algorithm)
        for architecture in ("tiny", "compact", "standard"):
            if algorithm == "ppo":
                specific = {"ppo": PPOConfig(architecture=architecture, rollout_steps=32,
                                             batch_size=32, epochs=2)}
            elif algorithm == "dqn":
                specific = {"dqn": DQNConfig(architecture=architecture, learning_starts=32,
                                             batch_size=32, replay_capacity=1000)}
            else:
                specific = {"tqc": TQCConfig(architecture=architecture, learning_starts=32,
                                             batch_size=32, replay_capacity=1000)}
            config = TrainingConfig(
                algorithm=algorithm, device=device.resolved, timesteps=args.steps,
                **specific,
            )
            env = ScaledTrainingReward(PolyTrackEnv(
                MockSimulatorTransport(), track_id=config.track_id,
                action_adapter=backend.action_adapter(config),
            ), config.reward_scale)
            try:
                model = backend.create_model(config, env, device.resolved)
                started = time.monotonic()
                model.learn(args.steps)
                seconds = time.monotonic() - started
                print(json.dumps({
                    "algorithm": algorithm, "architecture": architecture,
                    "device": device.resolved, "steps": model.num_timesteps,
                    "seconds": round(seconds, 3),
                    "tps": round(model.num_timesteps / seconds, 2),
                    "parameters": backend.parameter_counts(model),
                }), flush=True)
            finally:
                env.close()


if __name__ == "__main__":
    main()
