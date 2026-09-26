"""TQC with a forward-driving prior during replay-buffer warmup."""

from __future__ import annotations

import numpy as np
import torch
from sb3_contrib import TQC


class ForwardWarmupTQC(TQC):
    def __init__(
        self, *args,
        forward_warmup_fraction: float = 0.8,
        forward_warmup_steering_std: float = 0.45,
        forward_prior_initial: float = 0.0,
        forward_prior_steps: int = 0,
        **kwargs,
    ) -> None:
        self.forward_warmup_fraction = forward_warmup_fraction
        self.forward_warmup_steering_std = forward_warmup_steering_std
        self.forward_prior_initial = forward_prior_initial
        self.forward_prior_steps = forward_prior_steps
        self.actor_anchor_strength = 0.0
        self.actor_anchor_state = None
        super().__init__(*args, **kwargs)

    def anchor_actor(self, strength: float) -> None:
        """Keep fine-tuning close to a proven policy without freezing the actor."""
        if not 0.0 <= strength < 1.0:
            raise ValueError("actor anchor strength must be in [0, 1)")
        self.actor_anchor_strength = strength
        self.actor_anchor_state = [p.detach().cpu().clone() for p in self.actor.parameters()]

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        if self.actor_anchor_strength <= 0 or self.actor_anchor_state is None:
            return super().train(gradient_steps, batch_size)
        # One update at a time so the proximal pull applies after every actor step.
        for _ in range(gradient_steps):
            super().train(1, batch_size)
            with torch.no_grad():
                for parameter, anchor in zip(
                    self.actor.parameters(), self.actor_anchor_state, strict=True
                ):
                    parameter.lerp_(anchor.to(parameter.device), self.actor_anchor_strength)

    def forward_prior_strength(self, learning_starts: int) -> float:
        """Exploration-only longitudinal shift, zero after the configured decay."""
        if self.forward_prior_steps <= 0:
            return 0.0
        elapsed = max(0, self.num_timesteps - learning_starts)
        return self.forward_prior_initial * max(0.0, 1.0 - elapsed / self.forward_prior_steps)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if self.num_timesteps >= learning_starts:
            action, buffered = super()._sample_action(learning_starts, action_noise, n_envs)
            strength = self.forward_prior_strength(learning_starts)
            if strength > 0:
                buffered[:, 1] = np.clip(buffered[:, 1] + strength, -1.0, 1.0)
                action = self.policy.unscale_action(buffered)
            return action, buffered
        if self.forward_warmup_fraction == 0:
            return super()._sample_action(learning_starts, action_noise, n_envs)

        rng = self.action_space.np_random
        actions = np.array([self.action_space.sample() for _ in range(n_envs)])
        for action in actions:
            if rng.random() < self.forward_warmup_fraction:
                action[0] = np.clip(rng.normal(0.0, self.forward_warmup_steering_std), -1.0, 1.0)
                action[1] = rng.uniform(0.65, 1.0)
        scaled = self.policy.scale_action(actions)
        if action_noise is not None:
            scaled = np.clip(scaled + action_noise(), -1.0, 1.0)
        return self.policy.unscale_action(scaled), scaled
