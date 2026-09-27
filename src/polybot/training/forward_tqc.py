"""TQC with a forward-driving prior during replay-buffer warmup."""

from __future__ import annotations

import copy

import numpy as np
import torch
from sb3_contrib import TQC
from torch.nn import functional as F

from polybot.protocol import ROUTE_PROGRESS_FEATURE_INDEX


class ForwardWarmupTQC(TQC):
    def __init__(
        self, *args,
        forward_warmup_fraction: float = 0.8,
        forward_warmup_steering_std: float = 0.45,
        forward_prior_initial: float = 0.0,
        forward_prior_steps: int = 0,
        forward_guard_progress_ratio: float = 0.0,
        recovery_anchor_strength: float = 0.0,
        **kwargs,
    ) -> None:
        self.forward_warmup_fraction = forward_warmup_fraction
        self.forward_warmup_steering_std = forward_warmup_steering_std
        self.forward_prior_initial = forward_prior_initial
        self.forward_prior_steps = forward_prior_steps
        self.forward_guard_progress_ratio = forward_guard_progress_ratio
        self.recovery_anchor_strength = recovery_anchor_strength
        self.actor_anchor_strength = 0.0
        self.actor_anchor_state = None
        self.successful_trajectories: list[tuple[np.ndarray, np.ndarray]] = []
        self._success_rng = np.random.default_rng(kwargs.get("seed"))
        self.safe_actor_state: list[torch.Tensor] | None = None
        self.safe_training_state: dict | None = None
        self.safe_actor_progress = 0.0
        super().__init__(*args, **kwargs)

    @staticmethod
    def _cpu_snapshot(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: ForwardWarmupTQC._cpu_snapshot(item) for key, item in value.items()}
        if isinstance(value, list):
            return [ForwardWarmupTQC._cpu_snapshot(item) for item in value]
        if isinstance(value, tuple):
            return tuple(ForwardWarmupTQC._cpu_snapshot(item) for item in value)
        return copy.deepcopy(value)

    def mark_safe_actor(self, progress: float = 0.0) -> None:
        """Remember a policy and its learner state for collapse recovery."""
        self.safe_actor_state = [parameter.detach().cpu().clone()
                                 for parameter in self.actor.parameters()]
        self.safe_actor_progress = progress
        if self.recovery_anchor_strength > 0:
            self.actor_anchor_state = [parameter.clone() for parameter in self.safe_actor_state]
            self.actor_anchor_strength = self.recovery_anchor_strength
        # The demonstration pretrain has no useful critic yet. Capture the full
        # learner only after an actual episode has reached this progress.
        if progress > 0:
            self.safe_training_state = self._cpu_snapshot({
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor.optimizer.state_dict(),
                "critic_optimizer": self.critic.optimizer.state_dict(),
                "log_ent_coef": getattr(self, "log_ent_coef", None),
                "ent_coef_optimizer": (
                    self.ent_coef_optimizer.state_dict()
                    if getattr(self, "ent_coef_optimizer", None) is not None else None
                ),
            })

    def restore_safe_actor(self) -> bool:
        if self.safe_actor_state is None:
            return False
        with torch.no_grad():
            for parameter, safe in zip(
                self.actor.parameters(), self.safe_actor_state, strict=True
            ):
                parameter.copy_(safe.to(parameter.device))
        checkpoint = getattr(self, "safe_training_state", None)
        if checkpoint is None:
            self.actor.optimizer.state.clear()
        else:
            self.critic.load_state_dict(checkpoint["critic"])
            self.critic_target.load_state_dict(checkpoint["critic_target"])
            self.actor.optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic.optimizer.load_state_dict(checkpoint["critic_optimizer"])
            if checkpoint["log_ent_coef"] is not None:
                with torch.no_grad():
                    self.log_ent_coef.copy_(checkpoint["log_ent_coef"].to(self.device))
                self.ent_coef_optimizer.load_state_dict(checkpoint["ent_coef_optimizer"])
        if self.recovery_anchor_strength > 0:
            self.actor_anchor_state = [parameter.clone() for parameter in self.safe_actor_state]
            self.actor_anchor_strength = self.recovery_anchor_strength
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

    def _success_loss(self, batch_size: int) -> torch.Tensor:
        observations, actions = self.successful_trajectories[
            int(self._success_rng.integers(len(self.successful_trajectories)))
        ]
        indices = self._success_rng.integers(len(observations), size=min(batch_size, 64))
        obs = torch.as_tensor(observations[indices], device=self.device)
        target = torch.as_tensor(actions[indices], device=self.device)
        prediction = self.actor(obs, deterministic=True)
        return F.mse_loss(prediction, target)

    def _rehearse_success(self, batch_size: int) -> None:
        loss = self._success_loss(batch_size)
        self.actor.optimizer.zero_grad()
        loss.backward()
        self.actor.optimizer.step()
        if hasattr(self, "_logger"):
            self.logger.record("train/success_imitation_loss", loss.item())

    def _add_success_gradient(self, batch_size: int) -> None:
        loss = self._success_loss(batch_size)
        (2.0 * loss).backward()
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
        # Add imitation to the same actor optimizer step as TQC's reward gradient.
        hook = None
        if has_success:
            hook = self.actor.optimizer.register_step_pre_hook(
                lambda _optimizer, _args, _kwargs: self._add_success_gradient(batch_size)
            )
        try:
            for _ in range(gradient_steps):
                super().train(1, batch_size)
                if has_anchor:
                    with torch.no_grad():
                        for parameter, anchor in zip(
                            self.actor.parameters(), self.actor_anchor_state, strict=True
                        ):
                            parameter.lerp_(anchor.to(parameter.device),
                                            self.actor_anchor_strength)
        finally:
            if hook is not None:
                hook.remove()

    def forward_prior_strength(self, learning_starts: int) -> float:
        """Exploration-only longitudinal shift, zero after the configured decay."""
        if self.forward_prior_steps <= 0:
            return 0.0
        elapsed = max(0, self.num_timesteps - learning_starts)
        return self.forward_prior_initial * max(0.0, 1.0 - elapsed / self.forward_prior_steps)

    def _apply_forward_guard(self, buffered: np.ndarray) -> bool:
        if self.forward_guard_progress_ratio <= 0 or self._last_obs is None:
            return False
        progress = self._last_obs[:, ROUTE_PROGRESS_FEATURE_INDEX]
        guard = progress < self.forward_guard_progress_ratio
        if not np.any(guard):
            return False
        # Keep enough throttle to reach the demonstrated jump entry without early braking.
        buffered[guard, 1] = np.maximum(buffered[guard, 1], 0.5)
        return True

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        action, state = super().predict(
            observation, state=state, episode_start=episode_start,
            deterministic=deterministic,
        )
        if self.forward_guard_progress_ratio > 0:
            progress = np.asarray(observation)[..., ROUTE_PROGRESS_FEATURE_INDEX]
            guard = progress < self.forward_guard_progress_ratio
            action = np.asarray(action).copy()
            if action.ndim == 1:
                if bool(guard):
                    action[1] = max(float(action[1]), 0.5)
            else:
                action[guard, 1] = np.maximum(action[guard, 1], 0.5)
        return action, state

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        if self.num_timesteps >= learning_starts:
            action, buffered = super()._sample_action(learning_starts, action_noise, n_envs)
            strength = self.forward_prior_strength(learning_starts)
            if strength > 0:
                buffered[:, 1] = np.clip(buffered[:, 1] + strength, -1.0, 1.0)
            guard_applied = self._apply_forward_guard(buffered)
            if strength > 0 or guard_applied:
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
        if self._apply_forward_guard(scaled):
            actions = self.policy.unscale_action(scaled)
            return actions, scaled
        return self.policy.unscale_action(scaled), scaled
