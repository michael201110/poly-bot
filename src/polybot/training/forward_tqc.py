"""TQC with a forward-driving prior during replay-buffer warmup."""

from __future__ import annotations

import numpy as np
import torch
from sb3_contrib import TQC
from torch.nn import functional as F


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
        self.successful_trajectories: list[tuple[np.ndarray, np.ndarray]] = []
        self._success_rng = np.random.default_rng(kwargs.get("seed"))
        self.safe_actor_state: list[torch.Tensor] | None = None
        self.safe_actor_progress = 0.0
        super().__init__(*args, **kwargs)

    def mark_safe_actor(self, progress: float = 0.0) -> None:
        """Remember a demonstrated or finishing policy for collapse recovery."""
        self.safe_actor_state = [parameter.detach().cpu().clone()
                                 for parameter in self.actor.parameters()]
        self.safe_actor_progress = progress

    def restore_safe_actor(self) -> bool:
        if self.safe_actor_state is None:
            return False
        with torch.no_grad():
            for parameter, safe in zip(
                self.actor.parameters(), self.safe_actor_state, strict=True
            ):
                parameter.copy_(safe.to(parameter.device))
        self.actor.optimizer.state.clear()
        return True

    def remember_successful_trajectory(
        self, observations: np.ndarray, actions: np.ndarray
    ) -> None:
        """Keep completed-lap state/actions for actor rehearsal after the finish."""
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if observations.ndim != 2 or actions.shape != (len(observations), 2):
            raise ValueError("successful trajectory has invalid observation/action shape")
        if not len(observations):
            raise ValueError("successful trajectory must not be empty")
        self.successful_trajectories.append((observations.copy(), actions.copy()))
        self.successful_trajectories = self.successful_trajectories[-3:]

    def _rehearse_success(self, batch_size: int) -> None:
        observations, actions = self.successful_trajectories[
            int(self._success_rng.integers(len(self.successful_trajectories)))
        ]
        indices = self._success_rng.integers(len(observations), size=min(batch_size, 64))
        obs = torch.as_tensor(observations[indices], device=self.device)
        target = torch.as_tensor(actions[indices], device=self.device)
        prediction = self.actor(obs, deterministic=True)
        loss = F.mse_loss(prediction, target)
        self.actor.optimizer.zero_grad()
        loss.backward()
        self.actor.optimizer.step()
        if hasattr(self, "_logger"):
            self.logger.record("train/success_imitation_loss", loss.item())

    def anchor_actor(self, strength: float) -> None:
        """Keep fine-tuning close to a proven policy without freezing the actor."""
        if not 0.0 <= strength < 1.0:
            raise ValueError("actor anchor strength must be in [0, 1)")
        self.actor_anchor_strength = strength
        self.actor_anchor_state = [p.detach().cpu().clone() for p in self.actor.parameters()]

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        has_anchor = self.actor_anchor_strength > 0 and self.actor_anchor_state is not None
        has_success = bool(self.successful_trajectories)
        if not has_anchor and not has_success:
            return super().train(gradient_steps, batch_size)
        # Retain completed laps as the off-policy critic continues to change.
        for _ in range(gradient_steps):
            super().train(1, batch_size)
            if has_anchor:
                with torch.no_grad():
                    for parameter, anchor in zip(
                        self.actor.parameters(), self.actor_anchor_state, strict=True
                    ):
                        parameter.lerp_(anchor.to(parameter.device), self.actor_anchor_strength)
            if has_success:
                self._rehearse_success(batch_size)

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
