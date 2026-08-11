"""Widen a saved two-layer PPO model while preserving its exact behaviour."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from polybot.model_migration import policy_outputs, widen_actor_critic_policy


class _SpaceOnlyEnvironment(gym.Env[np.ndarray, np.ndarray]):
    def __init__(self, observation_space: gym.Space, action_space: gym.Space) -> None:
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action: np.ndarray):
        del action
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--width", type=int, default=128)
    args = parser.parse_args()
    if args.width < 1:
        parser.error("--width must be positive")

    source = PPO.load(args.source)
    environment = _SpaceOnlyEnvironment(source.observation_space, source.action_space)
    target = PPO(
        "MlpPolicy",
        environment,
        learning_rate=float(source.learning_rate),
        n_steps=source.n_steps,
        batch_size=source.batch_size,
        n_epochs=source.n_epochs,
        gamma=source.gamma,
        gae_lambda=source.gae_lambda,
        ent_coef=source.ent_coef,
        vf_coef=source.vf_coef,
        max_grad_norm=source.max_grad_norm,
        policy_kwargs={
            "net_arch": {
                "pi": [args.width, args.width],
                "vf": [args.width, args.width],
            }
        },
        verbose=0,
    )
    widen_actor_critic_policy(source.policy, target.policy)

    generator = torch.Generator().manual_seed(20260811)
    observations = torch.randn(
        (64, *source.observation_space.shape), generator=generator, dtype=torch.float32
    )
    source_logits, source_values = policy_outputs(source.policy, observations)
    target_logits, target_values = policy_outputs(target.policy, observations)
    torch.testing.assert_close(target_logits, source_logits, rtol=0, atol=1e-6)
    torch.testing.assert_close(target_values, source_values, rtol=0, atol=2e-4)
    maximum_logit_error = float((target_logits - source_logits).abs().max())
    maximum_value_error = float((target_values - source_values).abs().max())

    args.target.parent.mkdir(parents=True, exist_ok=True)
    target.save(args.target)
    print(
        f"Migrated {sum(p.numel() for p in source.policy.parameters()):,} -> "
        f"{sum(p.numel() for p in target.policy.parameters()):,} parameters; "
        f"max logit error {maximum_logit_error:.2g}, "
        f"max value error {maximum_value_error:.2g}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
